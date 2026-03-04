"""
CLI smoke test for saved checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dunhuang_dance_gen.models import validate_saved_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test saved SinMDM checkpoints.")
    parser.add_argument("--save_dir", default=str(PROJECT_ROOT / "save"), help="Directory containing saved runs.")
    parser.add_argument(
        "--report_dir",
        default=str(PROJECT_ROOT / "output_gui" / "model_reports"),
        help="Directory to store JSON/Markdown reports.",
    )
    parser.add_argument("--motion_length", type=float, default=1.0, help="Generated motion length for smoke tests.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-model timeout in seconds.")
    return parser


def render_markdown(results):
    lines = [
        "# Saved Model Smoke Test",
        "",
        "| Run | Dataset | Step | Result | Summary |",
        "|---|---|---:|---|---|",
    ]
    for item in results:
        record = item["record"]
        status = "PASS" if item["is_usable"] else "FAIL"
        lines.append(
            f"| {record['run_name']} | {record.get('dataset') or 'unknown'} | {record.get('step', -1)} | {status} | {item.get('summary') or ''} |"
        )
    return "\n".join(lines) + "\n"


def main():
    args = build_parser().parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    results = validate_saved_models(
        args.save_dir,
        latest_only=True,
        dataset_filter="bvh_general",
        python_executable=sys.executable,
        workdir=str(PROJECT_ROOT),
        output_root=str(PROJECT_ROOT / "output_gui" / "model_smoke"),
        motion_length=args.motion_length,
        timeout_seconds=args.timeout,
    )

    json_path = report_dir / "model_smoke_report.json"
    md_path = report_dir / "model_smoke_report.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")

    passed = sum(1 for item in results if item["is_usable"])
    print(f"Smoke-tested {len(results)} model(s): {passed} passed.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    for item in results:
        record = item["record"]
        status = "PASS" if item["is_usable"] else "FAIL"
        print(f"{record['run_name']}\t{status}\t{item['summary']}")

    sys.exit(0 if passed > 0 else 1)


if __name__ == "__main__":
    main()
