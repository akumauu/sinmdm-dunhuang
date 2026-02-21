"""
Demo Script for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 演示脚本

完整流程演示：加载 BVH -> 预处理 -> 验证 -> 后处理 -> 导出 -> 可视化
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from dunhuang_dance_gen.data.bvh_parser import BVHParser, load_bvh
from dunhuang_dance_gen.data.preprocess import DunhuangPreprocessor
from dunhuang_dance_gen.data.validator import DataValidator
from dunhuang_dance_gen.postprocess.smooth import MotionSmoother
from dunhuang_dance_gen.postprocess.constraints import PhysicalConstraints
from dunhuang_dance_gen.export.bvh_writer import BVHWriter
from dunhuang_dance_gen.visualize.viewer import MotionViewer


def demo_pipeline(bvh_path: str, output_dir: str = None):
    """
    演示完整的数据处理流程
    
    Args:
        bvh_path: 输入 BVH 文件路径
        output_dir: 输出目录
    """
    print("=" * 60)
    print("敦煌舞蹈动作生成系统 - 数据处理演示")
    print("=" * 60)
    
    bvh_path = Path(bvh_path)
    if output_dir is None:
        output_dir = bvh_path.parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载 BVH
    print("\n[1/6] 加载 BVH 文件...")
    try:
        data = load_bvh(str(bvh_path))
        print(f"  - 文件: {bvh_path.name}")
        print(f"  - 关节数: {data.num_joints}")
        print(f"  - 帧数: {data.num_frames}")
        print(f"  - FPS: {data.fps:.1f}")
        print(f"  - 时长: {data.duration:.2f}s")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    # 2. 数据验证
    print("\n[2/6] 数据验证...")
    validator = DataValidator()
    result = validator.validate(data)
    print(f"  - 有效性: {'✓ Valid' if result.is_valid else '✗ Invalid'}")
    if result.errors:
        for err in result.errors:
            print(f"    Error: {err}")
    if result.warnings:
        for warn in result.warnings[:3]:  # 只显示前3个警告
            print(f"    Warning: {warn}")
    
    quality = validator.compute_quality_score(data)
    print(f"  - 质量分数: {quality:.1f}/100")
    
    # 3. 预处理
    print("\n[3/6] 预处理...")
    preprocessor = DunhuangPreprocessor(target_fps=30.0)
    processed_data = preprocessor.process(
        data,
        resample=True,
        smooth=True,
        fix_outliers=True,
        normalize_scale=True
    )
    print(f"  - 处理后帧数: {processed_data.num_frames}")
    print(f"  - 处理后 FPS: {processed_data.fps:.1f}")
    
    # 4. 后处理
    print("\n[4/6] 后处理...")
    smoother = MotionSmoother(method='savgol', window_size=5)
    constraints = PhysicalConstraints()
    
    # 二次平滑
    smoothed_positions = smoother.smooth(processed_data.positions)
    smoothed_rotations = smoother.smooth(processed_data.rotations)
    
    # 物理约束
    final_positions, final_rotations = constraints.apply_all(
        smoothed_positions,
        smoothed_rotations,
        processed_data.joint_names
    )
    print("  - 平滑滤波: ✓")
    print("  - 物理约束: ✓")
    
    # 5. 导出
    print("\n[5/6] 导出 BVH...")
    writer = BVHWriter()
    output_path = output_dir / f"{bvh_path.stem}_processed.bvh"
    
    writer.write(
        filepath=str(output_path),
        joint_names=processed_data.joint_names,
        parent_indices=processed_data.parent_indices,
        offsets=processed_data.offsets,
        positions=final_positions,
        rotations=final_rotations,
        frame_time=processed_data.frame_time
    )
    print(f"  - 输出文件: {output_path}")
    
    # 6. 可视化（可选）
    print("\n[6/6] 可视化...")
    print("  提示: 调用 MotionViewer 查看动作...")
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    return {
        'input_data': data,
        'processed_data': processed_data,
        'output_path': str(output_path)
    }


def demo_visualization(bvh_path: str, frame_idx: int = 0):
    """
    演示可视化功能
    
    Args:
        bvh_path: BVH 文件路径
        frame_idx: 要显示的帧索引
    """
    print("加载动作数据...")
    data = load_bvh(bvh_path)
    
    print("计算关节位置...")
    # 简单版本：直接使用 offset 作为位置（实际需要前向运动学）
    # 这里仅作演示
    import numpy as np
    
    # 使用 offsets 构建初始骨架
    positions = data.offsets.copy()
    
    # 累加父节点位置
    for i, parent in enumerate(data.parent_indices):
        if parent >= 0:
            positions[i] += positions[parent]
    
    print("显示骨架...")
    viewer = MotionViewer()
    viewer.plot_skeleton(
        positions,
        data.parent_indices,
        data.joint_names,
        show_labels=True,
        title=f"Skeleton - {Path(bvh_path).name}"
    )


def main():
    parser = argparse.ArgumentParser(description='敦煌舞蹈动作生成系统演示')
    parser.add_argument('--bvh', type=str, help='输入 BVH 文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='显示可视化')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    
    args = parser.parse_args()
    
    if args.test:
        print("测试模式: 检查模块导入...")
        try:
            from dunhuang_dance_gen.data import BVHParser, DunhuangPreprocessor, DataValidator
            from dunhuang_dance_gen.postprocess import MotionSmoother, PhysicalConstraints
            from dunhuang_dance_gen.export import BVHWriter
            from dunhuang_dance_gen.visualize import MotionViewer
            from dunhuang_dance_gen.models import SinMDMWrapper
            print("✓ 所有模块导入成功!")
        except ImportError as e:
            print(f"✗ 导入失败: {e}")
        return
    
    if args.bvh:
        if args.visualize:
            demo_visualization(args.bvh)
        else:
            demo_pipeline(args.bvh, args.output)
    else:
        print("请指定 BVH 文件路径: python demo.py --bvh <path>")
        print("或运行测试模式: python demo.py --test")


if __name__ == "__main__":
    main()
