"""
敦煌舞数据集测试脚本
Test script for Dunhuang Dance dataset
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_bvh_parser():
    """测试 BVH 解析器"""
    from dunhuang_dance_gen.data.bvh_parser import load_bvh, BVHParser
    
    # 查找一个 BVH 文件
    dataset_root = Path(r"d:\sinMDM\敦煌舞三维动作数据集")
    bvh_files = list(dataset_root.rglob("*.bvh"))
    
    if not bvh_files:
        print("未找到 BVH 文件!")
        return False
    
    print(f"发现 {len(bvh_files)} 个 BVH 文件")
    
    # 测试第一个文件
    test_file = bvh_files[0]
    print(f"\n测试文件: {test_file.name}")
    
    try:
        data = load_bvh(str(test_file))
        print(f"  ✓ 加载成功!")
        print(f"    - 关节数: {data.num_joints}")
        print(f"    - 帧数: {data.num_frames}")
        print(f"    - FPS: {data.fps:.2f}")
        print(f"    - 时长: {data.duration:.2f}s")
        print(f"    - 关节名: {data.joint_names[:5]}...")
        return True
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validator():
    """测试数据验证器"""
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.data.validator import DataValidator
    
    dataset_root = Path(r"d:\sinMDM\敦煌舞三维动作数据集")
    bvh_files = list(dataset_root.rglob("*.bvh"))
    
    if not bvh_files:
        return False
    
    test_file = bvh_files[0]
    
    print(f"\n测试验证器: {test_file.name}")
    
    try:
        data = load_bvh(str(test_file))
        validator = DataValidator()
        
        result = validator.validate(data)
        print(f"  - 有效性: {'✓' if result.is_valid else '✗'}")
        
        quality = validator.compute_quality_score(data)
        print(f"  - 质量分数: {quality:.1f}/100")
        
        return True
    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def scan_dataset():
    """扫描整个数据集"""
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    
    dataset_root = Path(r"d:\sinMDM\敦煌舞三维动作数据集")
    
    categories = {}
    total_frames = 0
    total_duration = 0
    
    for bvh_file in dataset_root.rglob("*.bvh"):
        try:
            data = load_bvh(str(bvh_file))
            
            # 获取类别
            rel_path = bvh_file.relative_to(dataset_root)
            if len(rel_path.parts) >= 2:
                cat = rel_path.parts[1]
                if cat not in categories:
                    categories[cat] = {"count": 0, "frames": 0, "duration": 0}
                categories[cat]["count"] += 1
                categories[cat]["frames"] += data.num_frames
                categories[cat]["duration"] += data.duration
            
            total_frames += data.num_frames
            total_duration += data.duration
            
        except Exception as e:
            print(f"跳过 {bvh_file.name}: {e}")
    
    print("\n=== 数据集统计 ===")
    print(f"总文件数: {sum(c['count'] for c in categories.values())}")
    print(f"总帧数: {total_frames}")
    print(f"总时长: {total_duration/60:.1f} 分钟")
    
    print("\n类别统计:")
    for cat, stats in sorted(categories.items()):
        print(f"  {cat}: {stats['count']} 文件, {stats['duration']:.1f}s")


if __name__ == "__main__":
    print("=" * 50)
    print("敦煌舞数据集测试")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="扫描整个数据集")
    args = parser.parse_args()
    
    if args.scan:
        scan_dataset()
    else:
        test_bvh_parser()
        test_validator()
