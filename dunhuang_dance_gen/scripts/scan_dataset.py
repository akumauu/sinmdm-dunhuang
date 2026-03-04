"""
Dataset Scan Script - 敦煌舞蹈数据集统计扫描
生成数据集说明文档 (Markdown 格式)
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dunhuang_dance_gen.data.bvh_parser import load_bvh
from dunhuang_dance_gen.data.validator import DataValidator

DATASET_DIR = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
OUTPUT_FILE = PROJECT_ROOT / "docs" / "dataset_description.md"


def scan_dataset():
    """扫描数据集并生成统计"""
    validator = DataValidator()
    categories = {}
    
    for cat_dir in sorted(DATASET_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        
        cat_name = cat_dir.name
        entries = []
        
        for bvh_file in sorted(cat_dir.rglob("*.bvh")):
            try:
                data = load_bvh(str(bvh_file))
                result = validator.validate(data)
                quality = validator.compute_quality_score(data)
                
                entries.append({
                    'file': bvh_file.name,
                    'path': str(bvh_file.relative_to(DATASET_DIR)),
                    'joints': data.num_joints,
                    'frames': data.num_frames,
                    'fps': data.fps,
                    'duration': data.duration,
                    'valid': result.is_valid,
                    'quality': quality,
                    'warnings': len(result.warnings),
                })
            except Exception as e:
                entries.append({
                    'file': bvh_file.name,
                    'path': str(bvh_file.relative_to(DATASET_DIR)),
                    'error': str(e),
                })
        
        categories[cat_name] = entries
    
    return categories


def generate_markdown(categories):
    """生成 Markdown 文档"""
    total_files = sum(len(v) for v in categories.values())
    total_frames = sum(
        e.get('frames', 0) for entries in categories.values() for e in entries
    )
    total_duration = sum(
        e.get('duration', 0) for entries in categories.values() for e in entries
    )
    valid_count = sum(
        1 for entries in categories.values() for e in entries if e.get('valid', False)
    )
    
    lines = [
        "# 敦煌舞蹈三维动作数据集说明",
        "",
        "## 概览",
        "",
        "| 属性 | 值 |",
        "|------|-----|",
        f"| **数据来源** | 敦煌舞蹈三维动作采集 |",
        f"| **数据格式** | BVH (BioVision Hierarchy) |",
        f"| **舞蹈类别** | {len(categories)} 类 |",
        f"| **文件总数** | {total_files} 个 BVH 文件 |",
        f"| **总帧数** | {total_frames:,} 帧 |",
        f"| **总时长** | {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟) |",
        f"| **有效文件** | {valid_count}/{total_files} |",
        "",
        "## 舞蹈类别与文件详情",
        "",
    ]
    
    for cat_name, entries in categories.items():
        cat_duration = sum(e.get('duration', 0) for e in entries)
        lines.append(f"### {cat_name}")
        lines.append(f"")
        lines.append(f"共 {len(entries)} 个文件, 总时长 {cat_duration:.1f} 秒")
        lines.append(f"")
        lines.append("| 文件名 | 关节数 | 帧数 | FPS | 时长(s) | 质量分 | 状态 |")
        lines.append("|--------|--------|------|-----|---------|--------|------|")
        
        for e in entries:
            if 'error' in e:
                lines.append(f"| `{e['file']}` | - | - | - | - | - | ❌ {e['error'][:30]} |")
            else:
                status = "✅" if e['valid'] else "⚠️"
                lines.append(
                    f"| `{e['file']}` | {e['joints']} | {e['frames']} | "
                    f"{e['fps']:.0f} | {e['duration']:.1f} | {e['quality']:.0f} | {status} |"
                )
        lines.append("")
    
    lines.extend([
        "## 骨架结构",
        "",
        "所有 BVH 文件采用统一的 22 关节人体骨架拓扑:",
        "",
        "```",
        "Hips (ROOT)",
        "├── Spine → Spine1 → Spine2 → Neck → Head",
        "├── LeftUpLeg → LeftLeg → LeftFoot → LeftToeBase",
        "├── RightUpLeg → RightLeg → RightFoot → RightToeBase",
        "├── LeftShoulder → LeftArm → LeftForeArm → LeftHand",
        "└── RightShoulder → RightArm → RightForeArm → RightHand",
        "```",
        "",
        "## 数据处理说明",
        "",
        "- **帧率**: 原始数据统一为 30 FPS, 预处理可重采样到目标帧率",
        "- **坐标系**: Y-up, Z-forward, 右手坐标系",
        "- **旋转表示**: BVH 原始为欧拉角 (ZXY 顺序), 训练时转换为 6D 旋转表示",
        "- **数据维度**: 22 关节 × (3位移 + 6旋转) = 198 维输入",
        "",
        "---",
        f"*自动生成于数据集扫描脚本*",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("扫描敦煌舞蹈数据集...")
    categories = scan_dataset()
    
    md = generate_markdown(categories)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"数据集说明已生成: {OUTPUT_FILE}")
    print(f"共扫描 {len(categories)} 个类别, "
          f"{sum(len(v) for v in categories.values())} 个文件")
