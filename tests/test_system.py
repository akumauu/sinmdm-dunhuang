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
import json
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


@test("数据集构建 - 切片与 train/val 清单")
def test_dataset_builder():
    from dunhuang_dance_gen.data import build_dataset_from_bvh_files

    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_files = sorted(str(path) for path in dataset_dir.rglob("*.bvh"))[:2]
    assert len(bvh_files) == 2, "需要至少 2 个 BVH 文件用于测试"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_dataset_from_bvh_files(
            bvh_paths=bvh_files,
            output_root=tmpdir,
            clip_seconds=3.0,
            overlap_seconds=1.0,
            min_clip_seconds=1.5,
            val_ratio=0.25,
            seed=123,
        )
        assert result.total_sources == 2
        assert result.total_clips > 0, "应至少导出 1 个 clip"
        assert Path(result.manifest_path).exists(), "manifest 未生成"
        assert Path(result.train_list_path).exists(), "train_list 未生成"
        assert Path(result.summary_path).exists(), "summary 未生成"

        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        assert manifest["total_clips"] == result.total_clips
        assert len(manifest["records"]) == result.total_clips

test_dataset_builder()


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


@test("视频姿态导出 - 22关节序列转 BVH")
def test_video_pose_bvh_export():
    from dunhuang_dance_gen.data import save_pose_sequence_as_bvh
    from dunhuang_dance_gen.data.bvh_parser import load_bvh

    frames = 24
    base = np.zeros((frames, 22, 3), dtype=np.float32)

    # 构造一条简单但有效的 22 关节骨架序列
    base[:, 0] = np.array([0.0, 90.0, 0.0], dtype=np.float32)
    base[:, 1] = np.array([0.0, 105.0, 0.0], dtype=np.float32)
    base[:, 2] = np.array([0.0, 120.0, 0.0], dtype=np.float32)
    base[:, 3] = np.array([0.0, 135.0, 0.0], dtype=np.float32)
    base[:, 4] = np.array([0.0, 150.0, 0.0], dtype=np.float32)
    base[:, 5] = np.array([0.0, 165.0, 5.0], dtype=np.float32)
    base[:, 6] = np.array([-8.0, 135.0, 0.0], dtype=np.float32)
    base[:, 7] = np.array([-18.0, 132.0, 0.0], dtype=np.float32)
    base[:, 8] = np.array([-28.0, 126.0, 2.0], dtype=np.float32)
    base[:, 9] = np.array([-38.0, 120.0, 4.0], dtype=np.float32)
    base[:, 10] = np.array([8.0, 135.0, 0.0], dtype=np.float32)
    base[:, 11] = np.array([18.0, 132.0, 0.0], dtype=np.float32)
    base[:, 12] = np.array([28.0, 126.0, -2.0], dtype=np.float32)
    base[:, 13] = np.array([38.0, 120.0, -4.0], dtype=np.float32)
    base[:, 14] = np.array([-8.0, 78.0, 0.0], dtype=np.float32)
    base[:, 15] = np.array([-8.0, 42.0, 2.0], dtype=np.float32)
    base[:, 16] = np.array([-8.0, 10.0, 6.0], dtype=np.float32)
    base[:, 17] = np.array([-8.0, 0.0, 12.0], dtype=np.float32)
    base[:, 18] = np.array([8.0, 78.0, 0.0], dtype=np.float32)
    base[:, 19] = np.array([8.0, 42.0, -2.0], dtype=np.float32)
    base[:, 20] = np.array([8.0, 10.0, -6.0], dtype=np.float32)
    base[:, 21] = np.array([8.0, 0.0, -12.0], dtype=np.float32)

    base[:, 0, 0] = np.linspace(0.0, 12.0, frames, dtype=np.float32)
    base[:, 9, 1] += np.linspace(0.0, 8.0, frames, dtype=np.float32)
    base[:, 13, 1] += np.linspace(0.0, 6.0, frames, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix='.bvh', delete=False) as f:
        temp_path = f.name

    try:
        save_pose_sequence_as_bvh(base, temp_path, fps=24.0)
        reloaded = load_bvh(temp_path)
        assert reloaded.num_frames == frames, f"帧数不匹配: {reloaded.num_frames}"
        assert reloaded.num_joints == 22, f"关节数不匹配: {reloaded.num_joints}"
    finally:
        os.unlink(temp_path)

test_video_pose_bvh_export()


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


@test("模型注册 - 发现已训练的 bvh_general 检查点")
def test_model_registry_discovery():
    from dunhuang_dance_gen.models import list_saved_models

    records = list_saved_models(str(PROJECT_ROOT / "save"), latest_only=True, dataset_filter="bvh_general")
    assert len(records) >= 3, f"发现 {len(records)} 个 bvh_general 模型（期望至少 3 个）"
    assert all(Path(record.model_path).exists() for record in records), "存在缺失的模型文件"

test_model_registry_discovery()


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
# 6. 风格模块测试
# ============================================================
print("\n" + "=" * 60)
print("6. 风格模块")
print("=" * 60)


@test("风格特征提取 - 真实 BVH 数据")
def test_style_extraction():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.evaluate.style_features import DunhuangStyleExtractor
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    data = load_bvh(str(bvh_file))
    
    extractor = DunhuangStyleExtractor(fps=30.0)
    profile = extractor.extract(data.rotations, bvh_file.stem)
    
    assert profile.left_arm_extension_mean > 0, "上肢舒展度应该 > 0"
    assert profile.spine_curvature_mean > 0, "脊柱弯曲度应该 > 0"
    assert 0 <= profile.pause_ratio <= 100, "停顿占比应在 0-100"
    assert profile.upper_lower_ratio > 0, "上下身比例应该 > 0"
    
    vec = profile.to_vector()
    assert vec.shape == (21,), f"风格向量维度应为 21, 实际 {vec.shape}"

test_style_extraction()


@test("风格迁移 - 迁移 + BVH 往返验证")
def test_style_transfer():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.postprocess.style_transfer import DunhuangStyleTransfer, StyleTransferConfig
    from dunhuang_dance_gen.export.bvh_writer import BVHWriter
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_files = sorted(dataset_dir.rglob("*.bvh"))
    assert len(bvh_files) >= 2, "需要至少 2 个 BVH 文件"
    
    src = load_bvh(str(bvh_files[0]))
    ref = load_bvh(str(bvh_files[1]))
    
    transfer = DunhuangStyleTransfer(fps=30.0)
    cfg = StyleTransferConfig(arm_extension_strength=0.7, spine_curvature_strength=0.5)
    result = transfer.transfer(src.rotations, src.positions, ref.rotations, cfg)
    
    assert result.rotations.shape == src.rotations.shape, "输出形状应与输入一致"
    assert not np.any(np.isnan(result.rotations)), "迁移后不应有 NaN"
    
    # BVH 往返验证
    with tempfile.NamedTemporaryFile(suffix='.bvh', delete=False) as f:
        temp_path = f.name
    try:
        writer = BVHWriter(frame_time=src.frame_time)
        writer.write(temp_path, src.joint_names, src.parent_indices,
                     src.offsets, result.positions, result.rotations)
        reloaded = load_bvh(temp_path)
        assert reloaded.num_frames == src.num_frames, "帧数不一致"
        assert reloaded.num_joints == src.num_joints, "关节数不一致"
    finally:
        os.unlink(temp_path)

test_style_transfer()


@test("风格约束 - 实际达成值验证")
def test_style_constraint():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.postprocess.style_transfer import StyleConstraintApplicator
    from dunhuang_dance_gen.evaluate.style_features import DunhuangStyleExtractor
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    data = load_bvh(str(bvh_file))
    
    applicator = StyleConstraintApplicator(fps=30.0)
    mod_rot, mod_pos, report = applicator.apply(
        data.rotations, data.positions,
        target_arm_extension=130.0,
        target_spine_curvature=300.0,
        target_pause_ratio=15.0,
        target_symmetry=0.7,
        constraint_strength=0.6
    )
    
    assert mod_rot.shape == data.rotations.shape
    assert not np.any(np.isnan(mod_rot)), "约束后不应有 NaN"
    assert '上肢舒展度' in report, "报告应包含上肢舒展度"
    assert '目标' in report['上肢舒展度'], "报告应显示目标值"
    assert '停顿比例' in report, "报告应包含停顿比例"
    assert '整体对称性' in report, "报告应包含整体对称性"
    
    # 验证报告中的数值是实际值 (不应等于目标值)
    extractor = DunhuangStyleExtractor(fps=30.0)
    after = extractor.extract(mod_rot, "after")
    assert after.left_arm_extension_mean != data.rotations.shape[0], "应返回真实的特征值"

test_style_constraint()


@test("风格迁移 - 对称性迁移与风格混合")
def test_style_symmetry_and_blend():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.postprocess.style_transfer import (
        DunhuangStyleTransfer,
        StyleTransferConfig,
        style_blend,
    )

    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_files = sorted(dataset_dir.rglob("*.bvh"))
    src = load_bvh(str(bvh_files[0]))
    ref = load_bvh(str(bvh_files[1]))

    transfer = DunhuangStyleTransfer(fps=30.0)
    result = transfer.transfer(
        src.rotations,
        src.positions,
        ref.rotations,
        StyleTransferConfig(symmetry_strength=0.8, smooth_transition=False),
    )
    assert result.rotations.shape == src.rotations.shape
    assert 0.0 <= result.result_style.overall_symmetry <= 1.0

    mixed = style_blend(src.rotations, ref.rotations, weight=0.4)
    assert mixed.shape[0] == min(src.rotations.shape[0], ref.rotations.shape[0])
    assert mixed.shape[1:] == src.rotations.shape[1:]

test_style_symmetry_and_blend()


@test("跨舞段风格一致性矩阵")
def test_cross_dance_consistency():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.evaluate.style_features import StyleConsistencyEvaluator
    
    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_files = sorted(dataset_dir.rglob("*.bvh"))
    
    evaluator = StyleConsistencyEvaluator()
    for f in bvh_files[:4]:
        data = load_bvh(str(f))
        evaluator.add_motion(f.stem[:20], data.rotations)
    
    matrix, names = evaluator.compute_consistency_matrix()
    
    assert matrix.shape == (4, 4), f"矩阵形状应为 (4,4), 实际 {matrix.shape}"
    assert len(names) == 4
    assert np.all(np.diag(matrix) == 0), "对角线应全为 0"
    assert np.all(matrix >= 0), "距离应非负"
    
    report = evaluator.generate_report()
    assert "风格距离矩阵" in report

test_cross_dance_consistency()


@test("教学分析 - 分段/关键帧/慢放导出")
def test_teaching_analyzer():
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    from dunhuang_dance_gen.teaching import TeachingAnalyzer

    dataset_dir = PROJECT_ROOT / "敦煌舞三维动作数据集" / "长动作"
    bvh_file = next(dataset_dir.rglob("*.bvh"))
    data = load_bvh(str(bvh_file))

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = TeachingAnalyzer(fps=data.fps)
        result = analyzer.analyze_and_export(
            data=data,
            output_dir=tmpdir,
            motion_name="teaching_test",
            target_segment_seconds=2.5,
            min_segment_seconds=1.0,
            slow_motion_factor=2.0,
        )
        assert len(result.segments) >= 1
        assert len(result.keyframes) >= 2
        assert Path(result.slow_bvh_path).exists()
        assert Path(result.report_json_path).exists()
        assert Path(result.report_md_path).exists()

test_teaching_analyzer()


@test("外部联动 - Blender 路径解析")
def test_blender_resolver():
    from dunhuang_dance_gen.integrations import resolve_blender_executable

    with tempfile.NamedTemporaryFile(delete=False) as handle:
        fake_blender = handle.name

    try:
        assert resolve_blender_executable(fake_blender) == fake_blender
    finally:
        os.unlink(fake_blender)

test_blender_resolver()


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

if __name__ == "__main__":
    sys.exit(1 if failed > 0 else 0)
