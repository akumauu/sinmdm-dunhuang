"""
Functional Test Suite for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 功能测试脚本

覆盖所有核心模块的测试用例：
- 数据处理 (BVH 解析 / 预处理 / 验证)
- 后处理 (平滑 / 约束 / 管线)
- 导出 (BVH 写入 / 重新加载)
- 评估 (指标计算)
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 测试结果收集
test_results = []


def test(name):
    """测试装饰器"""
    def decorator(func):
        def wrapper():
            try:
                func()
                test_results.append(('✅', name, ''))
                print(f"  ✅ {name}")
            except Exception as e:
                test_results.append(('❌', name, str(e)))
                print(f"  ❌ {name}: {e}")
        return wrapper
    return decorator


# ============================================================
# 1. 数据处理模块测试
# ============================================================
print("\n" + "=" * 60)
print("1. 数据处理模块")
print("=" * 60)


@test("BVH 解析器 - 导入")
def test_bvh_parser_import():
    from dunhuang_dance_gen.data.bvh_parser import BVHParser, BVHData, load_bvh

test_bvh_parser_import()


@test("BVH 解析器 - 加载数据集")
def test_bvh_parser_load():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_files = list(dataset_dir.rglob("*.bvh"))
    
    assert len(bvh_files) > 0, "未找到 BVH 文件"
    
    loaded = 0
    for bvh_file in bvh_files[:3]:  # 测试前 3 个
        data = load_bvh(str(bvh_file))
        assert data.num_frames > 0, f"帧数为 0: {bvh_file.name}"
        assert data.num_joints > 0, f"关节数为 0: {bvh_file.name}"
        assert data.positions is not None, f"缺少位置数据: {bvh_file.name}"
        assert data.rotations is not None, f"缺少旋转数据: {bvh_file.name}"
        loaded += 1
    
    assert loaded >= 3, f"只成功加载了 {loaded} 个文件"

test_bvh_parser_load()


@test("预处理器 - 帧率重采样")
def test_preprocessor_resample():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.data.preprocess import DunhuangPreprocessor
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    data = load_bvh(str(bvh_file))
    
    preprocessor = DunhuangPreprocessor(target_fps=30.0)
    result = preprocessor.process(data, resample=True, smooth=False, 
                                   fix_outliers=False, normalize_scale=False)
    
    assert abs(result.fps - 30.0) < 1.0, f"帧率不正确: {result.fps}"

test_preprocessor_resample()


@test("预处理器 - 完整处理流程")
def test_preprocessor_full():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.data.preprocess import DunhuangPreprocessor
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    data = load_bvh(str(bvh_file))
    
    preprocessor = DunhuangPreprocessor(target_fps=30.0)
    result = preprocessor.process(data)
    
    assert result.num_frames > 0
    assert not np.any(np.isnan(result.positions)), "处理后有 NaN"
    assert not np.any(np.isnan(result.rotations)), "处理后有 NaN"

test_preprocessor_full()


@test("数据验证器 - 有效性检查")
def test_validator():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.data.validator import DataValidator
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    data = load_bvh(str(bvh_file))
    
    validator = DataValidator()
    result = validator.validate(data)
    
    assert result.is_valid, f"数据无效: {result.errors}"
    
    score = validator.compute_quality_score(data)
    assert 0 <= score <= 100, f"质量分数异常: {score}"

test_validator()


@test("数据验证器 - SinMDM 兼容性")
def test_validator_compatibility():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.data.validator import DataValidator
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    data = load_bvh(str(bvh_file))
    
    validator = DataValidator()
    result = validator.check_sinmdm_compatibility(data)
    
    assert result.is_valid, f"兼容性检查失败: {result.errors}"

test_validator_compatibility()


# ============================================================
# 2. 后处理模块测试
# ============================================================
print("\n" + "=" * 60)
print("2. 后处理模块")
print("=" * 60)


@test("平滑器 - Savitzky-Golay")
def test_smoother_savgol():
    from dunhuang_dance_gen.postprocess import MotionSmoother
    
    # 创建带噪声的测试数据
    np.random.seed(42)
    data = np.sin(np.linspace(0, 4 * np.pi, 100))
    noisy = data + np.random.randn(100) * 0.3
    
    smoother = MotionSmoother(method='savgol', window_size=7)
    smoothed = smoother.smooth(noisy)
    
    assert smoothed.shape == noisy.shape, "形状不一致"
    # 平滑后应更接近原始信号
    assert np.std(smoothed - data) < np.std(noisy - data), "平滑效果不佳"

test_smoother_savgol()


@test("平滑器 - 高斯滤波")
def test_smoother_gaussian():
    from dunhuang_dance_gen.postprocess import MotionSmoother
    
    np.random.seed(42)
    data = np.random.randn(100, 22, 3).astype(np.float32)
    
    smoother = MotionSmoother(method='gaussian', sigma=1.0)
    smoothed = smoother.smooth(data)
    
    assert smoothed.shape == data.shape
    assert smoothed.dtype == data.dtype

test_smoother_gaussian()


@test("平滑器 - 速度突变修复")
def test_smoother_velocity_fix():
    from dunhuang_dance_gen.postprocess import MotionSmoother
    
    # 创建包含突变的数据
    data = np.zeros(50)
    data[25] = 100.0  # 人为突变
    
    smoother = MotionSmoother()
    fixed = smoother.fix_velocity_spikes(data, threshold_factor=2.0)
    
    assert abs(fixed[25]) < 100.0, "突变未被修复"

test_smoother_velocity_fix()


@test("物理约束 - 关节限位")
def test_constraints_joint_limits():
    from dunhuang_dance_gen.postprocess import PhysicalConstraints
    
    constraints = PhysicalConstraints()
    
    # 创建超出范围的旋转数据
    rotations = np.random.randn(50, 5, 3).astype(np.float32) * 200  # 故意超范围
    joint_names = ['hip', 'knee', 'ankle', 'spine', 'elbow']
    
    fixed = constraints.apply_joint_limits(rotations, joint_names, soft=True)
    
    assert fixed.shape == rotations.shape

test_constraints_joint_limits()


@test("物理约束 - 地面穿透修正")
def test_constraints_ground():
    from dunhuang_dance_gen.postprocess import PhysicalConstraints
    
    constraints = PhysicalConstraints(ground_level=0.0)
    
    positions = np.array([[0, -5, 0], [0, 3, 0], [0, -1, 0]], dtype=np.float32)
    fixed = constraints.enforce_ground_contact(positions)
    
    assert np.all(fixed[:, 1] >= 0.0), "仍有穿透地面的点"

test_constraints_ground()


@test("后处理管线 - 完整流程")
def test_pipeline():
    from dunhuang_dance_gen.postprocess import PostProcessPipeline, PostProcessConfig
    
    config = PostProcessConfig(
        smooth_method='savgol',
        smooth_window=5,
        apply_joint_limits=True,
        stabilize_root=True,
        enforce_ground=True,
    )
    
    pipeline = PostProcessPipeline(config)
    
    np.random.seed(42)
    positions = np.random.randn(100, 3).astype(np.float32)
    rotations = np.random.randn(100, 10, 3).astype(np.float32) * 50
    joint_names = ['hip', 'spine', 'neck', 'shoulder_l', 'elbow_l',
                   'shoulder_r', 'elbow_r', 'knee_l', 'knee_r', 'ankle']
    
    result = pipeline.process(positions, rotations, joint_names)
    
    assert result.positions.shape == positions.shape
    assert result.rotations.shape == rotations.shape
    assert len(result.stats_before) > 0
    assert len(result.stats_after) > 0
    
    summary = result.summary()
    assert "后处理摘要" in summary

test_pipeline()


# ============================================================
# 3. 导出模块测试
# ============================================================
print("\n" + "=" * 60)
print("3. 导出模块")
print("=" * 60)


@test("BVH 写入器 - 写入并重新加载")
def test_bvh_writer():
    from dunhuang_dance_gen.export.bvh_writer import BVHWriter
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    
    # 创建测试数据
    joint_names = ['Hips', 'Spine', 'Head']
    parent_indices = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array([[0, 0, 0], [0, 10, 0], [0, 5, 0]], dtype=np.float32)
    positions = np.random.randn(30, 3).astype(np.float32)
    rotations = np.random.randn(30, 3, 3).astype(np.float32) * 20
    
    writer = BVHWriter(rotation_order='zxy')
    
    with tempfile.NamedTemporaryFile(suffix='.bvh', delete=False) as f:
        temp_path = f.name
    
    try:
        writer.write(temp_path, joint_names, parent_indices, offsets, 
                     positions, rotations)
        
        assert os.path.exists(temp_path), "文件未创建"
        assert os.path.getsize(temp_path) > 0, "文件为空"
        
        # 重新加载验证
        reloaded = load_bvh(temp_path)
        assert reloaded.num_frames == 30, f"帧数不匹配: {reloaded.num_frames}"
        assert reloaded.num_joints == 3, f"关节数不匹配: {reloaded.num_joints}"
    finally:
        os.unlink(temp_path)

test_bvh_writer()


@test("BVH 写入器 - 真实数据往返")
def test_bvh_roundtrip():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.export.bvh_writer import BVHWriter
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    
    original = load_bvh(str(bvh_file))
    
    writer = BVHWriter(frame_time=original.frame_time)
    
    with tempfile.NamedTemporaryFile(suffix='.bvh', delete=False) as f:
        temp_path = f.name
    
    try:
        writer.write(temp_path, original.joint_names, original.parent_indices,
                     original.offsets, original.positions, original.rotations)
        
        reloaded = load_bvh(temp_path)
        assert reloaded.num_frames == original.num_frames
        assert reloaded.num_joints == original.num_joints
    finally:
        os.unlink(temp_path)

test_bvh_roundtrip()


# ============================================================
# 4. 评估模块测试
# ============================================================
print("\n" + "=" * 60)
print("4. 评估模块")
print("=" * 60)


@test("评估器 - 基础指标计算")
def test_evaluator_basic():
    from dunhuang_dance_gen.evaluate import MotionEvaluator
    
    evaluator = MotionEvaluator(fps=30.0)
    
    np.random.seed(42)
    rotations = np.random.randn(100, 22, 3).astype(np.float32) * 30
    positions = np.random.randn(100, 3).astype(np.float32)
    joint_names = [f'joint_{i}' for i in range(22)]
    
    report = evaluator.evaluate(rotations, positions, joint_names, "test_motion")
    
    assert report.num_frames == 100
    assert report.motion_name == "test_motion"
    assert '角速度均值(°/帧)' in report.metrics
    assert '根节点平均速度' in report.metrics

test_evaluator_basic()


@test("评估器 - 分布对比")
def test_evaluator_comparison():
    from dunhuang_dance_gen.evaluate import MotionEvaluator
    
    evaluator = MotionEvaluator()
    
    np.random.seed(42)
    ref = np.random.randn(100, 10, 3).astype(np.float32) * 20
    gen = ref + np.random.randn(100, 10, 3).astype(np.float32) * 5
    pos = np.random.randn(100, 3).astype(np.float32)
    joint_names = [f'j{i}' for i in range(10)]
    
    report = evaluator.evaluate(gen, pos, joint_names, "compare_test",
                                 reference_rotations=ref)
    
    assert '均值距离' in report.metrics
    assert '方差距离' in report.metrics

test_evaluator_comparison()


@test("评估器 - Markdown 报告")
def test_evaluator_report():
    from dunhuang_dance_gen.evaluate import MotionEvaluator
    
    evaluator = MotionEvaluator()
    
    np.random.seed(42)
    rot = np.random.randn(50, 5, 3).astype(np.float32) * 10
    pos = np.random.randn(50, 3).astype(np.float32)
    names = ['hip', 'knee_l', 'knee_r', 'elbow_l', 'elbow_r']
    
    report = evaluator.evaluate(rot, pos, names, "report_test")
    md = report.to_markdown()
    
    assert "## 评估报告" in md
    assert "report_test" in md

test_evaluator_report()


@test("评估器 - 批量汇总表")
def test_evaluator_batch():
    from dunhuang_dance_gen.evaluate import MotionEvaluator
    
    evaluator = MotionEvaluator()
    
    np.random.seed(42)
    samples = []
    for _ in range(3):
        rot = np.random.randn(50, 5, 3).astype(np.float32) * 15
        pos = np.random.randn(50, 3).astype(np.float32)
        samples.append((rot, pos))
    
    names = ['hip', 'spine', 'neck', 'elbow', 'knee']
    reports = evaluator.batch_evaluate(samples, names, "batch")
    
    assert len(reports) == 3
    
    table = evaluator.summary_table(reports)
    assert "评估汇总表" in table
    assert "平均" in table

test_evaluator_batch()


# ============================================================
# 5. 集成测试
# ============================================================
print("\n" + "=" * 60)
print("5. 集成测试")
print("=" * 60)


@test("完整管线 - 加载→预处理→后处理→导出→评估")
def test_full_pipeline():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.data.preprocess import DunhuangPreprocessor
    from dunhuang_dance_gen.data.validator import DataValidator
    from dunhuang_dance_gen.postprocess import PostProcessPipeline, PostProcessConfig
    from dunhuang_dance_gen.export.bvh_writer import BVHWriter
    from dunhuang_dance_gen.evaluate import MotionEvaluator
    
    # 1. 加载
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    data = load_bvh(str(bvh_file))
    
    # 2. 验证
    validator = DataValidator()
    val_result = validator.validate(data)
    assert val_result.is_valid, f"数据无效: {val_result.errors}"
    
    # 3. 预处理
    preprocessor = DunhuangPreprocessor(target_fps=30.0)
    processed = preprocessor.process(data)
    
    # 4. 后处理
    config = PostProcessConfig(smooth_method='savgol', smooth_window=5)
    pipeline = PostProcessPipeline(config)
    result = pipeline.process(processed.positions, processed.rotations, 
                               processed.joint_names)
    
    # 5. 导出
    with tempfile.NamedTemporaryFile(suffix='.bvh', delete=False) as f:
        temp_path = f.name
    
    try:
        writer = BVHWriter(frame_time=processed.frame_time)
        writer.write(temp_path, processed.joint_names, processed.parent_indices,
                     processed.offsets, result.positions, result.rotations)
        
        assert os.path.exists(temp_path)
        
        # 6. 评估
        evaluator = MotionEvaluator(fps=processed.fps)
        report = evaluator.evaluate(
            result.rotations, result.positions,
            processed.joint_names, bvh_file.stem,
            reference_rotations=data.rotations
        )
        
        assert report.num_frames > 0
        assert len(report.metrics) > 0
    finally:
        os.unlink(temp_path)

test_full_pipeline()


@test("多舞段扫描 - 加载所有类别")
def test_multi_dance_scan():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.data.validator import DataValidator
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    categories = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    
    assert len(categories) >= 6, f"发现 {len(categories)} 个类别（期望至少 6 个）"
    
    validator = DataValidator()
    loaded = 0
    
    for cat_dir in dataset_dir.iterdir():
        if cat_dir.is_dir():
            bvh_files = list(cat_dir.rglob("*.bvh"))
            for bvh_file in bvh_files[:1]:  # 每类取第一个
                data = load_bvh(str(bvh_file))
                result = validator.validate(data)
                assert result.is_valid, f"{bvh_file.name}: {result.errors}"
                loaded += 1
    
    assert loaded >= 6, f"只成功加载了 {loaded} 个类别"

test_multi_dance_scan()


# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("测试报告汇总")
print("=" * 60)

total = len(test_results)
passed = sum(1 for r in test_results if r[0] == '✅')
failed = sum(1 for r in test_results if r[0] == '❌')

print(f"\n总计: {total} 项测试")
print(f"通过: {passed} ✅")
print(f"失败: {failed} ❌")
print(f"通过率: {passed/total*100:.1f}%\n")

if failed > 0:
    print("失败的测试:")
    for status, name, error in test_results:
        if status == '❌':
            print(f"  ❌ {name}: {error}")

# 输出 Markdown 格式测试表 (论文用)
print("\n" + "-" * 60)
print("Markdown 格式 (可直接用于论文):")
print("-" * 60)
print()
print("| 编号 | 测试项 | 结果 |")
print("|------|--------|------|")
for i, (status, name, _) in enumerate(test_results, 1):
    print(f"| T{i:02d} | {name} | {status} |")
print(f"| | **总计 {total} 项 · 通过率 {passed/total*100:.0f}%** | |")
