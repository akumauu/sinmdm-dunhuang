"""
Teaching-oriented motion analysis and export helpers.

This turns a BVH motion into a lightweight teaching package:
- auto segmentation
- keyframe discovery
- difficulty scoring
- slow-motion export for external playback tools
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..data.bvh_parser import BVHData
from ..evaluate.style_features import DunhuangStyleExtractor
from ..export.bvh_writer import BVHWriter


@dataclass
class TeachingSegment:
    index: int
    start_frame: int
    end_frame: int
    duration_sec: float
    energy_mean: float
    difficulty_score: float
    difficulty_level: str
    notes: str
    clip_path: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_sec": round(self.duration_sec, 4),
            "energy_mean": round(self.energy_mean, 6),
            "difficulty_score": round(self.difficulty_score, 2),
            "difficulty_level": self.difficulty_level,
            "notes": self.notes,
            "clip_path": self.clip_path,
        }


@dataclass
class TeachingAnalysisResult:
    motion_name: str
    segments: List[TeachingSegment]
    keyframes: List[int]
    slow_bvh_path: str
    report_json_path: str
    report_md_path: str
    average_difficulty: float
    style_summary: Dict[str, float] = field(default_factory=dict)

    def summary_markdown(self) -> str:
        lines = [
            "## ✅ 教学分析完成",
            "",
            "| 指标 | 值 |",
            "|------|----|",
            f"| 动作名 | {self.motion_name} |",
            f"| 分段数 | {len(self.segments)} |",
            f"| 关键帧数 | {len(self.keyframes)} |",
            f"| 平均难度 | {self.average_difficulty:.2f}/5 |",
            f"| 慢放 BVH | `{self.slow_bvh_path}` |",
            f"| JSON 报告 | `{self.report_json_path}` |",
            f"| Markdown 报告 | `{self.report_md_path}` |",
        ]
        if self.keyframes:
            preview = ", ".join(str(idx) for idx in self.keyframes[:12])
            lines.extend(["", f"关键帧: {preview}"])
        if self.segments:
            lines.extend(["", "### 分段建议", "", "| 段 | 帧范围 | 时长(s) | 难度 | 说明 |", "|---|---|---:|---|---|"])
            for segment in self.segments:
                lines.append(
                    f"| {segment.index + 1} | {segment.start_frame}-{segment.end_frame} | "
                    f"{segment.duration_sec:.2f} | {segment.difficulty_level}({segment.difficulty_score:.1f}) | "
                    f"{segment.notes} |"
                )
        return "\n".join(lines)


class TeachingAnalyzer:
    """Generate a lightweight teaching package from a BVH motion."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.style_extractor = DunhuangStyleExtractor(fps=fps)

    def analyze_and_export(
        self,
        data: BVHData,
        output_dir: str,
        motion_name: str = "motion",
        target_segment_seconds: float = 3.0,
        min_segment_seconds: float = 1.5,
        slow_motion_factor: float = 2.0,
    ) -> TeachingAnalysisResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fps = max(float(data.fps), 1.0)
        energy = self._motion_energy(data)
        segments = self._segment_motion(
            data,
            energy,
            target_segment_seconds=target_segment_seconds,
            min_segment_seconds=min_segment_seconds,
        )
        keyframes = self._select_keyframes(energy, segments, data.num_frames)

        clips_dir = output_path / "segments"
        clips_dir.mkdir(parents=True, exist_ok=True)
        writer = BVHWriter(frame_time=data.frame_time)

        for segment in segments:
            clip_data = self._slice_data(data, segment.start_frame, segment.end_frame)
            clip_path = clips_dir / f"{motion_name}_segment_{segment.index + 1:02d}.bvh"
            writer.write_from_bvhdata(str(clip_path), clip_data)
            segment.clip_path = str(clip_path)

        slow_bvh_path = str(output_path / f"{motion_name}_slow.bvh")
        slow_data = self._slow_motion_copy(data, slow_motion_factor)
        writer.write_from_bvhdata(slow_bvh_path, slow_data)

        style_profile = self.style_extractor.extract(data.rotations, motion_name)
        average_difficulty = float(np.mean([segment.difficulty_score for segment in segments])) if segments else 1.0

        report_payload = {
            "motion_name": motion_name,
            "average_difficulty": round(average_difficulty, 4),
            "keyframes": keyframes,
            "slow_bvh_path": slow_bvh_path,
            "style_summary": {
                "pause_ratio": round(style_profile.pause_ratio, 4),
                "upper_lower_ratio": round(style_profile.upper_lower_ratio, 4),
                "overall_symmetry": round(style_profile.overall_symmetry, 4),
                "arm_extension_mean": round(
                    0.5 * (style_profile.left_arm_extension_mean + style_profile.right_arm_extension_mean),
                    4,
                ),
            },
            "segments": [segment.to_dict() for segment in segments],
        }

        report_json_path = str(output_path / "teaching_report.json")
        report_md_path = str(output_path / "teaching_report.md")
        with open(report_json_path, "w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, ensure_ascii=False, indent=2)

        result = TeachingAnalysisResult(
            motion_name=motion_name,
            segments=segments,
            keyframes=keyframes,
            slow_bvh_path=slow_bvh_path,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            average_difficulty=average_difficulty,
            style_summary=report_payload["style_summary"],
        )
        Path(report_md_path).write_text(result.summary_markdown(), encoding="utf-8")
        return result

    def _motion_energy(self, data: BVHData) -> np.ndarray:
        if data.num_frames <= 1:
            return np.zeros((data.num_frames,), dtype=np.float32)
        rot_delta = np.diff(data.rotations.reshape(data.num_frames, -1), axis=0)
        pos_delta = np.diff(data.positions, axis=0)
        rot_energy = np.linalg.norm(rot_delta, axis=1)
        pos_energy = np.linalg.norm(pos_delta, axis=1) * 0.25
        energy = rot_energy + pos_energy
        return np.concatenate([energy[:1], energy]).astype(np.float32)

    def _segment_motion(
        self,
        data: BVHData,
        energy: np.ndarray,
        target_segment_seconds: float,
        min_segment_seconds: float,
    ) -> List[TeachingSegment]:
        fps = max(float(data.fps), 1.0)
        target_frames = max(2, int(round(target_segment_seconds * fps)))
        min_frames = max(2, int(round(min_segment_seconds * fps)))

        if data.num_frames <= target_frames:
            return [self._build_segment(0, 0, data.num_frames, energy, data.rotations)]

        smooth_window = max(3, min(target_frames // 2, data.num_frames))
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones((smooth_window,), dtype=np.float32) / smooth_window
        smooth_energy = np.convolve(energy, kernel, mode="same")

        boundaries = [0]
        cursor = target_frames
        while cursor < data.num_frames - min_frames:
            search_start = max(boundaries[-1] + min_frames, cursor - target_frames // 3)
            search_end = min(data.num_frames - min_frames, cursor + target_frames // 3)
            if search_end <= search_start:
                boundary = min(data.num_frames - min_frames, boundaries[-1] + target_frames)
            else:
                local_idx = int(np.argmin(smooth_energy[search_start:search_end]))
                boundary = search_start + local_idx
            if boundary - boundaries[-1] < min_frames:
                boundary = min(data.num_frames - min_frames, boundaries[-1] + target_frames)
            if boundary <= boundaries[-1]:
                break
            boundaries.append(boundary)
            cursor = boundary + target_frames
        boundaries.append(data.num_frames)

        segments: List[TeachingSegment] = []
        for index in range(len(boundaries) - 1):
            start = boundaries[index]
            end = boundaries[index + 1]
            if end - start < 2:
                continue
            segments.append(self._build_segment(index, start, end, energy[start:end], data.rotations[start:end]))
        return segments or [self._build_segment(0, 0, data.num_frames, energy, data.rotations)]

    def _build_segment(
        self,
        index: int,
        start_frame: int,
        end_frame: int,
        energy_slice: np.ndarray,
        rotation_slice: np.ndarray,
    ) -> TeachingSegment:
        local_fps = max(self.fps, 1.0)
        profile = self.style_extractor.extract(rotation_slice, f"segment_{index + 1}")
        energy_mean = float(np.mean(energy_slice)) if len(energy_slice) else 0.0
        amplitude = float(np.std(rotation_slice))
        asymmetry = 1.0 - float(profile.overall_symmetry)
        pause_penalty = 1.0 - min(max(profile.pause_ratio / 100.0, 0.0), 1.0)

        score = 1.0 + min(
            4.0,
            1.6 * min(energy_mean / 80.0, 1.0)
            + 1.2 * min(amplitude / 90.0, 1.0)
            + 0.8 * pause_penalty
            + 0.8 * min(max(asymmetry, 0.0), 1.0),
        )
        if score < 2.0:
            level = "入门"
        elif score < 3.2:
            level = "初级"
        elif score < 4.2:
            level = "进阶"
        else:
            level = "高阶"

        notes = []
        if pause_penalty > 0.7:
            notes.append("连续性强")
        if profile.overall_symmetry < 0.45:
            notes.append("左右配合复杂")
        if amplitude > 70.0:
            notes.append("幅度较大")
        if not notes:
            notes.append("可作为拆解练习段")

        return TeachingSegment(
            index=index,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            duration_sec=float((end_frame - start_frame) / local_fps),
            energy_mean=energy_mean,
            difficulty_score=float(score),
            difficulty_level=level,
            notes="，".join(notes),
        )

    def _select_keyframes(self, energy: np.ndarray, segments: List[TeachingSegment], num_frames: int) -> List[int]:
        candidates = {0, max(0, num_frames - 1)}
        for segment in segments:
            candidates.add(segment.start_frame)
            candidates.add(max(segment.start_frame, segment.end_frame - 1))
            local_energy = energy[segment.start_frame:segment.end_frame]
            if len(local_energy) > 0:
                peak = int(np.argmax(local_energy)) + segment.start_frame
                candidates.add(peak)
        keyframes = sorted(idx for idx in candidates if 0 <= idx < num_frames)
        return keyframes[:12]

    def _slice_data(self, data: BVHData, start_frame: int, end_frame: int) -> BVHData:
        return BVHData(
            joint_names=list(data.joint_names),
            parent_indices=data.parent_indices.copy(),
            offsets=data.offsets.copy(),
            positions=data.positions[start_frame:end_frame].copy(),
            rotations=data.rotations[start_frame:end_frame].copy(),
            frame_time=float(data.frame_time),
            num_frames=int(end_frame - start_frame),
            num_joints=int(data.num_joints),
            rotation_order=str(data.rotation_order),
        )

    def _slow_motion_copy(self, data: BVHData, slow_motion_factor: float) -> BVHData:
        factor = max(1.0, float(slow_motion_factor))
        return BVHData(
            joint_names=list(data.joint_names),
            parent_indices=data.parent_indices.copy(),
            offsets=data.offsets.copy(),
            positions=data.positions.copy(),
            rotations=data.rotations.copy(),
            frame_time=float(data.frame_time * factor),
            num_frames=int(data.num_frames),
            num_joints=int(data.num_joints),
            rotation_order=str(data.rotation_order),
        )
