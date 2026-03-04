"""
Dataset organization utilities for BVH-based training assets.

This module fills the "dataset construction" gap from the thesis scope:
- discover BVH files from a source directory
- slice long motions into fixed-length clips
- create deterministic train/val splits
- export manifests and ready-to-train clip files
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .bvh_parser import BVHData, load_bvh
from ..export.bvh_writer import BVHWriter


@dataclass
class DatasetClipRecord:
    """Metadata for one exported BVH clip."""

    source_path: str
    clip_path: str
    split: str
    category: str
    clip_name: str
    start_frame: int
    end_frame: int
    num_frames: int
    duration_sec: float
    fps: float

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "clip_path": self.clip_path,
            "split": self.split,
            "category": self.category,
            "clip_name": self.clip_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "num_frames": self.num_frames,
            "duration_sec": round(self.duration_sec, 4),
            "fps": round(self.fps, 4),
        }


@dataclass
class DatasetBuildResult:
    """Result bundle for a dataset build run."""

    output_root: str
    manifest_path: str
    summary_path: str
    train_list_path: str
    val_list_path: str
    total_sources: int
    total_clips: int
    train_clips: int
    val_clips: int
    skipped_sources: List[str] = field(default_factory=list)
    records: List[DatasetClipRecord] = field(default_factory=list)

    def summary_markdown(self) -> str:
        lines = [
            "## ✅ 数据集构建完成",
            "",
            "| 指标 | 值 |",
            "|------|----|",
            f"| 源文件数 | {self.total_sources} |",
            f"| 导出 clip 数 | {self.total_clips} |",
            f"| 训练集 clip | {self.train_clips} |",
            f"| 验证集 clip | {self.val_clips} |",
            f"| 跳过源文件 | {len(self.skipped_sources)} |",
            f"| Manifest | `{self.manifest_path}` |",
            f"| 训练列表 | `{self.train_list_path}` |",
            f"| 验证列表 | `{self.val_list_path}` |",
        ]
        if self.skipped_sources:
            lines.extend(["", "### 跳过文件"])
            lines.extend([f"- `{Path(path).name}`" for path in self.skipped_sources[:10]])
            if len(self.skipped_sources) > 10:
                lines.append(f"- 其余 {len(self.skipped_sources) - 10} 个文件已写入 summary")
        return "\n".join(lines)


def discover_bvh_files(source_root: str) -> List[str]:
    """Discover all BVH files under a source root or return the single file."""
    root = Path(source_root)
    if not root.exists():
        return []
    if root.is_file():
        return [str(root)] if root.suffix.lower() == ".bvh" else []
    return sorted(str(path) for path in root.rglob("*.bvh"))


def _slice_motion_windows(
    num_frames: int,
    clip_frames: int,
    stride_frames: int,
    min_clip_frames: int,
) -> List[Tuple[int, int]]:
    if num_frames < min_clip_frames:
        return []
    if num_frames <= clip_frames:
        return [(0, num_frames)]

    windows: List[Tuple[int, int]] = []
    start = 0
    clip_frames = max(clip_frames, min_clip_frames)
    stride_frames = max(1, stride_frames)

    while start < num_frames:
        end = min(start + clip_frames, num_frames)
        if end - start >= min_clip_frames:
            windows.append((start, end))
        if end >= num_frames:
            break
        start += stride_frames

    if windows and windows[-1][1] < num_frames:
        tail_start = max(0, num_frames - clip_frames)
        if num_frames - tail_start >= min_clip_frames:
            tail = (tail_start, num_frames)
            if tail not in windows:
                windows.append(tail)

    return windows


def _slice_bvh_data(data: BVHData, start_frame: int, end_frame: int) -> BVHData:
    positions = data.positions[start_frame:end_frame].copy()
    rotations = data.rotations[start_frame:end_frame].copy()
    num_frames = int(end_frame - start_frame)
    return BVHData(
        joint_names=list(data.joint_names),
        parent_indices=data.parent_indices.copy(),
        offsets=data.offsets.copy(),
        positions=positions,
        rotations=rotations,
        frame_time=float(data.frame_time),
        num_frames=num_frames,
        num_joints=int(data.num_joints),
        rotation_order=str(data.rotation_order),
    )


def _category_for_path(path: Path) -> str:
    if path.parent == path:
        return "uncategorized"
    return path.parent.name or "uncategorized"


def _deterministic_split_token(source_path: str, start_frame: int, end_frame: int, seed: int) -> float:
    token = f"{source_path}|{start_frame}|{end_frame}|{seed}".encode("utf-8")
    digest = hashlib.sha1(token).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def build_dataset_from_bvh_files(
    bvh_paths: Sequence[str],
    output_root: str,
    clip_seconds: float = 4.0,
    overlap_seconds: float = 1.0,
    min_clip_seconds: float = 2.0,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetBuildResult:
    """
    Slice BVH files into train/val clips and export a manifest.

    The split is deterministic per clip so repeated runs remain stable.
    """
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root_path / "dataset_manifest.json"
    summary_path = output_root_path / "dataset_summary.md"
    train_list_path = output_root_path / "train_list.txt"
    val_list_path = output_root_path / "val_list.txt"

    writer = BVHWriter()
    records: List[DatasetClipRecord] = []
    skipped_sources: List[str] = []

    for source_path in bvh_paths:
        source_file = Path(source_path)
        try:
            data = load_bvh(str(source_file))
        except Exception:
            skipped_sources.append(str(source_file))
            continue

        fps = max(float(data.fps), 1.0)
        clip_frames = max(2, int(round(clip_seconds * fps)))
        overlap_frames = max(0, int(round(overlap_seconds * fps)))
        min_clip_frames = max(2, int(round(min_clip_seconds * fps)))
        stride_frames = max(1, clip_frames - overlap_frames)
        windows = _slice_motion_windows(data.num_frames, clip_frames, stride_frames, min_clip_frames)

        if not windows:
            skipped_sources.append(str(source_file))
            continue

        category = _category_for_path(source_file)
        base_name = source_file.stem

        for clip_index, (start_frame, end_frame) in enumerate(windows):
            split_value = _deterministic_split_token(str(source_file), start_frame, end_frame, seed)
            split = "val" if split_value < max(0.0, min(1.0, val_ratio)) else "train"

            clip_data = _slice_bvh_data(data, start_frame, end_frame)
            clip_name = f"{base_name}__clip{clip_index:03d}_f{start_frame:05d}_{end_frame:05d}.bvh"
            clip_dir = output_root_path / "clips" / split / category
            clip_path = clip_dir / clip_name
            writer.write_from_bvhdata(str(clip_path), clip_data)

            record = DatasetClipRecord(
                source_path=str(source_file),
                clip_path=str(clip_path),
                split=split,
                category=category,
                clip_name=clip_name,
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                num_frames=int(clip_data.num_frames),
                duration_sec=float(clip_data.duration),
                fps=fps,
            )
            records.append(record)

    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]

    manifest = {
        "output_root": str(output_root_path),
        "total_sources": len(list(bvh_paths)),
        "total_clips": len(records),
        "train_clips": len(train_records),
        "val_clips": len(val_records),
        "skipped_sources": skipped_sources,
        "records": [record.to_dict() for record in records],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    train_list_path.write_text(
        "\n".join(record.clip_path for record in train_records),
        encoding="utf-8",
    )
    val_list_path.write_text(
        "\n".join(record.clip_path for record in val_records),
        encoding="utf-8",
    )

    result = DatasetBuildResult(
        output_root=str(output_root_path),
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
        train_list_path=str(train_list_path),
        val_list_path=str(val_list_path),
        total_sources=len(list(bvh_paths)),
        total_clips=len(records),
        train_clips=len(train_records),
        val_clips=len(val_records),
        skipped_sources=skipped_sources,
        records=records,
    )
    summary_path.write_text(result.summary_markdown(), encoding="utf-8")
    return result


def build_dataset_from_root(
    source_root: str,
    output_root: str,
    clip_seconds: float = 4.0,
    overlap_seconds: float = 1.0,
    min_clip_seconds: float = 2.0,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetBuildResult:
    """Convenience wrapper around root discovery + dataset export."""
    bvh_paths = discover_bvh_files(source_root)
    return build_dataset_from_bvh_files(
        bvh_paths=bvh_paths,
        output_root=output_root,
        clip_seconds=clip_seconds,
        overlap_seconds=overlap_seconds,
        min_clip_seconds=min_clip_seconds,
        val_ratio=val_ratio,
        seed=seed,
    )
