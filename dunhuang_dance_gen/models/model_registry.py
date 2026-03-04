"""
Saved-model registry and smoke-test utilities.

This module centralizes discovery and validation of trained checkpoints so the
GUI and CLI can work from the same model inventory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SavedModelRecord:
    run_name: str
    model_path: str
    args_path: Optional[str]
    dataset: Optional[str]
    arch: Optional[str]
    num_joints: Optional[Any]
    sin_path: Optional[str]
    model_count: int
    step: int

    @property
    def is_bvh_general(self) -> bool:
        return self.dataset == "bvh_general"

    @property
    def label(self) -> str:
        dataset = self.dataset or "unknown"
        return f"{self.run_name} [{dataset}] step={self.step}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_cross_platform_path(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return value
    if os.path.exists(value):
        return value

    win_match = re.match(r"^([a-zA-Z]):[\\/](.*)$", value)
    if not win_match:
        return value

    drive_letter = win_match.group(1).lower()
    relative_path = win_match.group(2).replace("\\", "/")
    candidate = f"/mnt/{drive_letter}/{relative_path}"
    if os.path.exists(candidate):
        return candidate
    return value


def _parse_step(model_path: Path) -> int:
    match = re.search(r"model(\d+)\.pt$", model_path.name)
    if not match:
        return -1
    return int(match.group(1))


def list_saved_models(
    save_dir: str,
    latest_only: bool = True,
    dataset_filter: Optional[str] = None,
) -> List[SavedModelRecord]:
    save_path = Path(save_dir)
    if not save_path.exists():
        return []

    records: List[SavedModelRecord] = []
    for run_dir in sorted([p for p in save_path.iterdir() if p.is_dir()]):
        args_path = run_dir / "args.json"
        args: Dict[str, Any] = {}
        if args_path.exists():
            try:
                args = json.loads(args_path.read_text(encoding="utf-8"))
            except Exception:
                args = {}

        models = sorted(run_dir.glob("model*.pt"), key=_parse_step)
        if not models:
            continue

        selected = [models[-1]] if latest_only else models
        for model_path in selected:
            record = SavedModelRecord(
                run_name=run_dir.name,
                model_path=str(model_path),
                args_path=str(args_path) if args_path.exists() else None,
                dataset=args.get("dataset"),
                arch=args.get("arch"),
                num_joints=args.get("num_joints"),
                sin_path=_normalize_cross_platform_path(args.get("sin_path")),
                model_count=len(models),
                step=_parse_step(model_path),
            )
            if dataset_filter and record.dataset != dataset_filter:
                continue
            records.append(record)

    records.sort(key=lambda item: (item.dataset != "bvh_general", item.run_name, item.step))
    return records


def _last_non_empty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return ""


def smoke_test_model(
    model_path: str,
    python_executable: Optional[str] = None,
    workdir: Optional[str] = None,
    output_root: Optional[str] = None,
    motion_length: float = 1.0,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    model_path_obj = Path(model_path)
    run_name = model_path_obj.parent.name
    output_base = Path(output_root) if output_root else PROJECT_ROOT / "output_gui" / "model_smoke"
    output_dir = output_base / run_name

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_executable or sys.executable,
        "-m",
        "sample.generate",
        "--model_path",
        str(model_path_obj),
        "--output_dir",
        str(output_dir),
        "--num_samples",
        "1",
        "--motion_length",
        str(float(motion_length)),
    ]

    run = None
    summary = ""
    timeout_hit = False
    try:
        run = subprocess.run(
            cmd,
            cwd=workdir or str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
        summary = _last_non_empty_line(run.stdout or "")
    except subprocess.TimeoutExpired as exc:
        timeout_hit = True
        summary = _last_non_empty_line(exc.stdout or "") or "timeout"

    sample_bvh = output_dir / "sample00.bvh"
    preview_candidates = sorted(output_dir.glob("*.mp4")) + sorted(output_dir.glob("*.gif"))
    preview_asset = str(preview_candidates[0]) if preview_candidates else None
    is_usable = (
        not timeout_hit
        and run is not None
        and run.returncode == 0
        and sample_bvh.exists()
        and sample_bvh.stat().st_size > 0
    )

    return {
        "run_name": run_name,
        "model_path": str(model_path_obj),
        "output_dir": str(output_dir),
        "sample_bvh": str(sample_bvh) if sample_bvh.exists() else None,
        "preview_asset": preview_asset,
        "is_usable": is_usable,
        "returncode": 124 if timeout_hit else (run.returncode if run is not None else 1),
        "summary": summary,
        "timeout": timeout_hit,
    }


def validate_saved_models(
    save_dir: str,
    latest_only: bool = True,
    dataset_filter: Optional[str] = "bvh_general",
    python_executable: Optional[str] = None,
    workdir: Optional[str] = None,
    output_root: Optional[str] = None,
    motion_length: float = 1.0,
    timeout_seconds: int = 180,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for record in list_saved_models(save_dir, latest_only=latest_only, dataset_filter=dataset_filter):
        result = smoke_test_model(
            record.model_path,
            python_executable=python_executable,
            workdir=workdir,
            output_root=output_root,
            motion_length=motion_length,
            timeout_seconds=timeout_seconds,
        )
        result["record"] = record.to_dict()
        results.append(result)
    return results
