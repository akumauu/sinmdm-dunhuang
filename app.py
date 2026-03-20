"""
Dunhuang Dance Motion Generation System - Gradio Web Interface
敦煌舞蹈动作生成系统 - 桌面端交互界面

基于 Gradio 构建的一体化操作界面，包含：
- 数据管理：BVH 文件加载与信息展示
- 模型训练：超参数配置与训练启动
- 生成预览：参数控制与骨架动画预览
- 导出设置：后处理配置与 BVH 导出
- 3D 可视化：交互式骨骼查看与对比回放
- 项目管理：新建/保存/打开生成项目
- 视频姿态估计：MediaPipe 骨骼提取接口
"""

import os
import sys
import json
import glob
import shlex
import subprocess
import tempfile
import zipfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    print("请先安装 Gradio: pip install gradio")
    sys.exit(1)

from dunhuang_dance_gen.data import extract_video_to_bvh, build_dataset_from_root
from dunhuang_dance_gen.data.bvh_parser import BVHParser, load_bvh
from dunhuang_dance_gen.models import list_saved_models, validate_saved_models
from dunhuang_dance_gen.postprocess import (
    MotionSmoother,
    PhysicalConstraints,
    PostProcessPipeline,
    PostProcessConfig,
    style_blend,
)
from dunhuang_dance_gen.export.bvh_writer import BVHWriter
from dunhuang_dance_gen.evaluate.metrics import MotionEvaluator
from dunhuang_dance_gen.teaching import TeachingAnalyzer
from dunhuang_dance_gen.integrations import launch_blender_with_file

# 3D 可视化模块
try:
    from dunhuang_dance_gen.visualize.skeleton_viewer import (
        visualize_bvh, compare_bvh, render_skeleton_frame, render_comparison
    )
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 项目管理
PROJECT_DIR = str(PROJECT_ROOT / "projects")
os.makedirs(PROJECT_DIR, exist_ok=True)

# ============================================================
# 全局配置
# ============================================================
DATASET_DIR = str(PROJECT_ROOT / "敦煌舞三维动作数据集")
SAVE_DIR = str(PROJECT_ROOT / "save")
OUTPUT_DIR = str(PROJECT_ROOT / "output_gui")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_REPORT_DIR = str(Path(OUTPUT_DIR) / "model_reports")
os.makedirs(MODEL_REPORT_DIR, exist_ok=True)
DATASET_BUILD_DIR = str(Path(OUTPUT_DIR) / "dataset_builds")
os.makedirs(DATASET_BUILD_DIR, exist_ok=True)
TEACHING_OUTPUT_DIR = str(Path(OUTPUT_DIR) / "teaching")
os.makedirs(TEACHING_OUTPUT_DIR, exist_ok=True)


def _timestamp() -> str:
    """生成稳定的时间戳，用于输出目录命名。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _shell_join(cmd) -> str:
    """为终端展示构造可复制命令。"""
    return shlex.join([str(part) for part in cmd])


def _coerce_local_path(value) -> str:
    """Normalize Gradio file/path values to a usable local path string."""
    if value is None:
        return ""
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _zip_directory(source_dir: str, zip_path: str) -> str:
    source = Path(source_dir)
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(source)))
    return str(target)


def build_dataset_package(
    source_root: str,
    clip_seconds: float,
    overlap_seconds: float,
    min_clip_seconds: float,
    val_ratio: float,
):
    source_root = (source_root or "").strip()
    if not source_root:
        return "❌ 请输入 BVH 数据根目录", None
    if not Path(source_root).exists():
        return f"❌ 路径不存在: `{source_root}`", None

    run_dir = Path(DATASET_BUILD_DIR) / f"dataset_{_timestamp()}"
    try:
        result = build_dataset_from_root(
            source_root=source_root,
            output_root=str(run_dir),
            clip_seconds=float(clip_seconds),
            overlap_seconds=float(overlap_seconds),
            min_clip_seconds=float(min_clip_seconds),
            val_ratio=float(val_ratio),
        )
    except Exception as exc:
        return f"❌ 数据集构建失败: {exc}", None

    if result.total_clips <= 0:
        return (
            "❌ 未导出任何 clip。请检查源目录中的 BVH 文件，或降低最小时长参数。",
            None,
        )

    package_path = _zip_directory(str(run_dir), str(run_dir.with_suffix(".zip")))
    return result.summary_markdown(), package_path


def generate_teaching_package(
    bvh_path: str,
    segment_seconds: float,
    min_segment_seconds: float,
    slow_motion_factor: float,
):
    bvh_path = _coerce_local_path(bvh_path)
    if not bvh_path:
        return "❌ 请选择 BVH 文件", None, None
    if not Path(bvh_path).exists():
        return f"❌ 文件不存在: `{bvh_path}`", None, None

    try:
        data = load_bvh(bvh_path)
        out_dir = Path(TEACHING_OUTPUT_DIR) / f"{Path(bvh_path).stem}_{_timestamp()}"
        analyzer = TeachingAnalyzer(fps=data.fps)
        result = analyzer.analyze_and_export(
            data=data,
            output_dir=str(out_dir),
            motion_name=Path(bvh_path).stem,
            target_segment_seconds=float(segment_seconds),
            min_segment_seconds=float(min_segment_seconds),
            slow_motion_factor=float(slow_motion_factor),
        )
        package_path = _zip_directory(str(out_dir), str(out_dir.with_suffix(".zip")))
        return result.summary_markdown(), package_path, result.slow_bvh_path
    except Exception as exc:
        return f"❌ 教学分析失败: {exc}", None, None


def open_bvh_in_blender(preferred_bvh, fallback_bvh, blender_executable: str):
    preferred_path = _coerce_local_path(preferred_bvh)
    fallback_path = _coerce_local_path(fallback_bvh)
    target_path = preferred_path or fallback_path
    if not target_path:
        return "❌ 请先选择或生成一个 BVH 文件"

    ok, message = launch_blender_with_file(
        target_path,
        blender_executable.strip() if blender_executable else None,
    )
    if ok:
        return f"✅ {message}"
    return f"❌ {message}"


# ============================================================
# Tab 1: 数据管理
# ============================================================
def load_bvh_info(filepath: str):
    """加载 BVH 文件并返回信息摘要"""
    if not filepath or not os.path.exists(filepath):
        return "❌ 请选择有效的 BVH 文件", "", None
    
    try:
        data = load_bvh(filepath)
        
        info = f"""## 📋 BVH 文件信息

| 属性 | 值 |
|------|-----|
| **文件名** | `{Path(filepath).name}` |
| **关节数** | {data.num_joints} |
| **帧数** | {data.num_frames} |
| **帧率** | {data.fps:.1f} FPS |
| **时长** | {data.duration:.2f} 秒 |
| **帧时间** | {data.frame_time:.6f} 秒 |

### 骨架关节列表
"""
        for i, name in enumerate(data.joint_names):
            parent = data.parent_indices[i]
            parent_name = data.joint_names[parent] if parent >= 0 else "ROOT"
            info += f"- `{name}` ← `{parent_name}`\n"
        
        # 统计信息
        stats = f"""### 数据统计
| 维度 | 位置范围 (min/max) | 旋转范围 (min/max) |
|------|-------------------|-------------------|
| X | {data.positions[:, 0].min():.2f} / {data.positions[:, 0].max():.2f} | {data.rotations[:, :, 0].min():.2f} / {data.rotations[:, :, 0].max():.2f} |
| Y | {data.positions[:, 1].min():.2f} / {data.positions[:, 1].max():.2f} | {data.rotations[:, :, 1].min():.2f} / {data.rotations[:, :, 1].max():.2f} |
| Z | {data.positions[:, 2].min():.2f} / {data.positions[:, 2].max():.2f} | {data.rotations[:, :, 2].min():.2f} / {data.rotations[:, :, 2].max():.2f} |
"""
        info += stats
        
        # 生成骨架预览图
        preview_path = _generate_skeleton_preview(data)
        
        return info, f"✅ 成功加载: {data.num_joints} 关节, {data.num_frames} 帧, {data.duration:.1f}秒", preview_path
        
    except Exception as e:
        return f"❌ 加载失败: {str(e)}", "", None


def _generate_skeleton_preview(data) -> Optional[str]:
    """生成骨架预览图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 取中间帧
        mid_frame = data.num_frames // 2
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 通过正向运动学计算关节世界坐标 (简化版: 使用偏移量)
        positions = data.offsets.copy()
        # 从根节点累积偏移
        for i in range(len(data.joint_names)):
            parent = data.parent_indices[i]
            if parent >= 0:
                positions[i] = positions[parent] + data.offsets[i]
        
        # 绘制关节和骨骼
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='#2196F3', s=60, alpha=0.9, zorder=5)
        
        for i, parent in enumerate(data.parent_indices):
            if parent >= 0:
                ax.plot([positions[i, 0], positions[parent, 0]],
                       [positions[i, 1], positions[parent, 1]],
                       [positions[i, 2], positions[parent, 2]],
                       color='#1565C0', linewidth=2.5, alpha=0.8)
        
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title(f'骨架结构预览 ({data.num_joints} 关节)', fontsize=12, fontweight='bold')
        ax.view_init(elev=15, azim=45)
        
        # 等比例
        max_range = max(
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ) / 2.0
        mid = positions.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        plt.tight_layout()
        preview_path = os.path.join(OUTPUT_DIR, "skeleton_preview.png")
        fig.savefig(preview_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        return preview_path
    except Exception as e:
        print(f"预览图生成失败: {e}")
        return None


def scan_bvh_files():
    """扫描数据集目录中的 BVH 文件"""
    files = []
    for scan_root in [DATASET_DIR, SAVE_DIR, OUTPUT_DIR]:
        if not os.path.exists(scan_root):
            continue
        for root, dirs, filenames in os.walk(scan_root):
            for f in filenames:
                if f.endswith('.bvh'):
                    files.append(os.path.join(root, f))
    
    return sorted(set(files))


# ============================================================
# Tab 2: 模型训练
# ============================================================
def get_available_bvh_for_training():
    """获取可用于训练的 BVH 文件列表"""
    files = []
    for scan_root in [DATASET_DIR, OUTPUT_DIR]:
        if not os.path.exists(scan_root):
            continue
        for root, dirs, filenames in os.walk(scan_root):
            for f in filenames:
                if f.endswith('.bvh'):
                    files.append(os.path.join(root, f))
    return sorted(set(files))


def build_train_command(
    bvh_path: str,
    save_dir: str,
    num_steps: int,
    save_interval: int,
    arch: str,
    lr_gamma: float,
    gen_during_training: bool,
):
    """构建训练命令"""
    cmd = [
        sys.executable, "-m", "train.train_sinmdm",
        "--arch", arch,
        "--dataset", "bvh_general",
        "--sin_path", os.path.abspath(bvh_path),
        "--save_dir", os.path.abspath(save_dir),
        "--lr_method", "ExponentialLR",
        "--lr_gamma", str(lr_gamma),
        "--num_steps", str(num_steps),
        "--save_interval", str(save_interval),
        "--use_scale_shift_norm",
        "--use_checkpoint",
        "--overwrite",
    ]
    if gen_during_training:
        cmd.append("--gen_during_training")
    
    return cmd


def start_training(bvh_path, num_steps, save_interval, arch, lr_gamma, gen_during):
    """启动训练并返回后台进程信息。"""
    if not bvh_path:
        return "❌ 请先选择训练数据", ""

    if not os.path.exists(bvh_path):
        return "❌ 训练数据文件不存在", ""

    run_name = f"{Path(bvh_path).stem}_{arch}_{_timestamp()}"
    save_dir = os.path.join(SAVE_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")

    cmd = build_train_command(
        bvh_path,
        save_dir,
        int(num_steps),
        int(save_interval),
        arch,
        float(lr_gamma),
        gen_during,
    )
    cmd_str = _shell_join(cmd)

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception as e:
        return f"❌ 启动训练失败: {e}", cmd_str

    info = f"""## 🚀 训练已启动

```bash
{cmd_str}
```

### 训练配置
| 参数 | 值 |
|------|-----|
| **数据路径** | `{Path(bvh_path).name}` |
| **输出目录** | `{save_dir}` |
| **架构** | {arch} |
| **训练步数** | {num_steps} |
| **保存间隔** | {save_interval} |
| **学习率衰减** | {lr_gamma} |
| **训练中生成** | {'✅' if gen_during else '❌'} |
| **进程 PID** | {process.pid} |
| **日志文件** | `{log_path}` |

> 训练进程已在当前 WSL 环境后台启动。可以直接查看 `train.log` 跟踪进度。
"""
    return info, cmd_str


# ============================================================
# Tab 3: 生成预览
# ============================================================
def get_available_models(dataset_filter: Optional[str] = "bvh_general"):
    """获取已训练模型列表"""
    try:
        return [record.model_path for record in list_saved_models(SAVE_DIR, latest_only=True, dataset_filter=dataset_filter)]
    except Exception:
        models = []
        save_path = Path(SAVE_DIR)
        if not save_path.exists():
            return models
        for d in save_path.iterdir():
            if d.is_dir():
                pts = list(d.glob("model*.pt"))
                if pts:
                    models.append(str(sorted(pts)[-1]))
        return models


def _render_model_validation_markdown(results, json_path: str, md_path: str) -> str:
    total = len(results)
    passed = sum(1 for item in results if item["is_usable"])

    lines = [
        "## 🩺 模型体检完成",
        "",
        f"- 扫描模型数: **{total}**",
        f"- 可用模型数: **{passed}**",
        f"- JSON 报告: `{json_path}`",
        f"- Markdown 报告: `{md_path}`",
        "",
        "| Run | Step | Result | 摘要 |",
        "|---|---:|---|---|",
    ]

    for item in results:
        record = item["record"]
        status = "PASS" if item["is_usable"] else "FAIL"
        lines.append(
            f"| {record['run_name']} | {record.get('step', -1)} | {status} | {item.get('summary') or ''} |"
        )

    usable = [item["record"]["model_path"] for item in results if item["is_usable"]]
    if usable:
        lines.extend([
            "",
            "### 推荐默认模型",
            f"- `{usable[0]}`",
        ])

    return "\n".join(lines)


def scan_saved_models_ui():
    """批量体检 save 目录中的主线模型。"""
    try:
        results = validate_saved_models(
            SAVE_DIR,
            latest_only=True,
            dataset_filter="bvh_general",
            python_executable=sys.executable,
            workdir=str(PROJECT_ROOT),
            output_root=str(Path(OUTPUT_DIR) / "model_smoke"),
            motion_length=1.0,
            timeout_seconds=180,
        )
    except Exception as e:
        return f"❌ 模型体检失败: {e}", gr.update(choices=get_available_models(), value=None)

    json_path = str(Path(MODEL_REPORT_DIR) / "model_smoke_report.json")
    md_path = str(Path(MODEL_REPORT_DIR) / "model_smoke_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    markdown = _render_model_validation_markdown(results, json_path, md_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    usable_models = [item["record"]["model_path"] for item in results if item["is_usable"]]
    fallback_choices = get_available_models()
    choices = usable_models if usable_models else fallback_choices
    value = choices[0] if choices else None

    return markdown, gr.update(choices=choices, value=value)


def build_generate_command(model_path, output_dir, num_samples, motion_length, seed, diversity):
    """构建生成命令"""
    cmd = [
        sys.executable, "-m", "sample.generate",
        "--model_path", os.path.abspath(model_path),
        "--output_dir", os.path.abspath(output_dir),
        "--num_samples", str(int(num_samples)),
        "--motion_length", str(float(motion_length)),
        "--noise_scale", str(float(diversity)),
    ]
    if seed >= 0:
        cmd.append("--seed")
        cmd.append(str(int(seed)))
    
    return cmd


def generate_motion(model_path, num_samples, motion_length, seed, diversity):
    """执行生成并返回结果预览。"""
    if not model_path:
        return "❌ 请选择模型", "", None, None

    if not os.path.exists(model_path):
        return "❌ 模型文件不存在", "", None, None

    output_dir = os.path.join(
        OUTPUT_DIR,
        "generated",
        f"{Path(model_path).stem}_{_timestamp()}",
    )
    os.makedirs(output_dir, exist_ok=True)

    cmd = build_generate_command(model_path, output_dir, num_samples, motion_length, seed, diversity)
    cmd_str = _shell_join(cmd)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "").strip()
        preview_info = f"""## ❌ 生成失败

```bash
{cmd_str}
```

```text
{error_text}
```
"""
        return preview_info, cmd_str, None, None

    output_path = Path(output_dir)
    generated_bvh = sorted(output_path.glob("sample*.bvh"))
    generated_mp4 = sorted(output_path.glob("*.mp4"))
    generated_gif = sorted(output_path.glob("*.gif"))
    generated_npy = sorted(output_path.glob("results*.npy"))

    preview_info = f"""## 🎭 生成完成

```bash
{cmd_str}
```

### 参数设置
| 参数 | 值 |
|------|-----|
| **模型** | `{Path(model_path).name}` |
| **输出目录** | `{output_dir}` |
| **样本数** | {int(num_samples)} |
| **时长** | {motion_length} 秒 |
| **种子** | {'随机' if seed < 0 else int(seed)} |
| **多样性系数** | {float(diversity):.2f} |
"""

    if generated_bvh:
        preview_info += "\n### 本次生成结果\n"
        for f in generated_bvh[:5]:
            preview_info += f"- 📄 `{f.name}`\n"
    if generated_npy:
        preview_info += f"\n- 📦 结果数组: `{generated_npy[0].name}`\n"

    video_path = None
    preview_asset = None
    if generated_mp4:
        video_path = str(generated_mp4[0])
        preview_asset = video_path
    elif generated_gif:
        preview_asset = str(generated_gif[0])

    if preview_asset:
        preview_info += f"\n- 🎞️ 预览文件: `{Path(preview_asset).name}`\n"
    
    return preview_info, cmd_str, video_path, preview_asset


# ============================================================
# Tab 4: 导出与后处理
# ============================================================
def apply_postprocess(
    bvh_path: str,
    smooth_method: str,
    smooth_window: int,
    fix_spikes: bool,
    apply_limits: bool,
    stabilize: bool,
):
    """对 BVH 文件应用后处理"""
    if not bvh_path or not os.path.exists(bvh_path):
        return "❌ 请选择有效的 BVH 文件", None
    
    try:
        # 加载 BVH
        data = load_bvh(bvh_path)
        
        # 配置后处理
        config = PostProcessConfig(
            smooth_method=smooth_method,
            smooth_window=int(smooth_window),
            fix_velocity_spikes=fix_spikes,
            apply_joint_limits=apply_limits,
            stabilize_root=stabilize,
        )
        
        pipeline = PostProcessPipeline(config)
        result = pipeline.process(
            data.positions, 
            data.rotations, 
            data.joint_names
        )
        
        # 评估
        evaluator = MotionEvaluator(fps=data.fps)
        eval_before = evaluator.evaluate(
            data.rotations, data.positions, data.joint_names,
            motion_name="处理前"
        )
        eval_after = evaluator.evaluate(
            result.rotations, result.positions, data.joint_names,
            motion_name="处理后"
        )
        
        # 导出处理后的 BVH
        output_name = Path(bvh_path).stem + "_postprocessed.bvh"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        writer = BVHWriter(rotation_order='zxy', frame_time=data.frame_time)
        writer.write(
            output_path,
            data.joint_names,
            data.parent_indices,
            data.offsets,
            result.positions,
            result.rotations,
            data.frame_time
        )
        
        # 生成报告
        report = f"""## ✅ 后处理完成

### 配置
| 参数 | 值 |
|------|-----|
| 平滑方法 | {smooth_method} |
| 窗口大小 | {smooth_window} |
| 速度修正 | {'✅' if fix_spikes else '❌'} |
| 关节限位 | {'✅' if apply_limits else '❌'} |
| 根节点稳定 | {'✅' if stabilize else '❌'} |

### 处理前后对比

{evaluator.summary_table([eval_before, eval_after])}

### 输出文件
📄 `{output_path}`

> 可直接导入 Blender 验证效果
"""
        return report, output_path
        
    except Exception as e:
        return f"❌ 处理失败: {str(e)}", None


# ============================================================
# 主界面构建
# ============================================================

def scan_export_final_bvh():
    export_dir = r'D:\sinMDM\export_final\export'
    files = []
    if os.path.exists(export_dir):
        for root, _, filenames in os.walk(export_dir):
            for f in filenames:
                if f.endswith('.bvh'):
                    files.append(os.path.join(root, f))
    return sorted(files)

def scan_export_final_mp4():
    scan_dirs = [
        r'D:\sinMDM\export_final\export',
        r'D:\sinMDM\export_final\video',
    ]
    files = []
    for export_dir in scan_dirs:
        if os.path.exists(export_dir):
            for root, _, filenames in os.walk(export_dir):
                for f in filenames:
                    if f.endswith('.mp4'):
                        files.append(os.path.join(root, f))
    return sorted(set(files))


def check_system_environment():
    """检测并汇报系统环境状态"""
    import platform
    checks = []
    checks.append("## 🔧 系统环境检测报告\n")
    checks.append(f"| 项目 | 状态 |")
    checks.append(f"|------|------|")

    # OS
    checks.append(f"| **操作系统** | {platform.system()} {platform.release()} ({platform.architecture()[0]}) |")
    checks.append(f"| **Python** | {sys.version.split()[0]} (`{sys.executable}`) |")

    # CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else 'N/A'
        checks.append(f"| **PyTorch** | {torch.__version__} |")
        checks.append(f"| **CUDA 可用** | {'✅ ' + gpu_name if cuda_ok else '❌ 不可用 (CPU模式)'} |")
    except ImportError:
        checks.append(f"| **PyTorch** | ❌ 未安装 |")

    # Gradio
    checks.append(f"| **Gradio** | {gr.__version__} |")

    # WSL
    try:
        r = subprocess.run(["wsl", "--status"], capture_output=True, text=True, timeout=8)
        wsl_ok = r.returncode == 0
        checks.append(f"| **WSL 子系统** | {'✅ 可用' if wsl_ok else '⚠️ 异常'} |")
    except Exception:
        checks.append(f"| **WSL 子系统** | ⚠️ 未检测到 |")

    # Shell
    try:
        r = subprocess.run(["powershell", "-Command", "echo ok"], capture_output=True, text=True, timeout=5)
        checks.append(f"| **PowerShell** | {'✅ 可用' if r.returncode == 0 else '❌'} |")
    except Exception:
        checks.append(f"| **PowerShell** | ⚠️ 未检测到 |")

    # Blender
    try:
        r = subprocess.run(["blender", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            ver = r.stdout.strip().split('\n')[0]
            checks.append(f"| **Blender** | ✅ {ver} |")
        else:
            checks.append(f"| **Blender** | ⚠️ 未在 PATH |")
    except Exception:
        checks.append(f"| **Blender** | ⚠️ 未在 PATH |")

    # 数据目录
    export_dir = r'D:\sinMDM\export_final\export'
    video_dir = r'D:\sinMDM\export_final\video'
    n_bvh = len(scan_export_final_bvh())
    n_mp4 = len(scan_export_final_mp4())
    checks.append(f"| **BVH 资产库** | {'✅' if n_bvh > 0 else '❌'} {n_bvh} 个文件 |")
    checks.append(f"| **MP4 视频库** | {'✅' if n_mp4 > 0 else '❌'} {n_mp4} 个文件 |")
    checks.append(f"| **数据集目录** | {'✅ 存在' if os.path.exists(DATASET_DIR) else '❌ 缺失'} |")
    checks.append(f"| **模型存档** | {'✅ 存在' if os.path.exists(SAVE_DIR) else '❌ 缺失'} |")

    # 3D可视化
    checks.append(f"| **3D 可视化 (Plotly)** | {'✅ 可用' if HAS_PLOTLY else '❌ 缺失'} |")

    checks.append("\n---\n")
    checks.append(f"*检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    return "\n".join(checks)

def create_app():
    """创建 Gradio 应用"""
    
    # 预扫描文件
    bvh_files = scan_bvh_files()
    model_files = get_available_models()
    
    with gr.Blocks(
        title="敦煌舞蹈动作生成系统",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 10px; }
        .tab-content { min-height: 500px; }
        .gradio-container { max-width: 1400px !important; }
        .env-panel { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); border-radius: 12px; padding: 16px; color: #e0e0e0; }
        .advanced-demo-card { border: 1px solid rgba(100,149,237,0.3); border-radius: 10px; padding: 12px; background: rgba(100,149,237,0.05); }
        """
    ) as app:
        
        gr.Markdown("""
        # 🎭 敦煌舞蹈动作生成系统
        ### 基于 SinMDM 的单序列扩散模型 · 西北民族大学本科毕业设计
        """, elem_classes="main-title")

        # ==========================================
        # 环境检测面板 (折叠)
        # ==========================================
        with gr.Accordion("🔧 系统环境检测 · 点击展开", open=False):
            env_check_btn = gr.Button("🔍 执行环境自检", variant="secondary")
            env_report = gr.Markdown("点击上方按钮进行环境检测...")
            env_check_btn.click(check_system_environment, outputs=[env_report])

        with gr.Tabs():
            # ==========================================
            # 模块一：【🎯 动作生成与展示】
            # ==========================================
            with gr.TabItem("🎯 动作生成与展示", id="generation_demo"):
                gr.Markdown("### 🎭 零样本/单样本动作生成及高质量渲染成果展示")
                
                with gr.Accordion("1. 成果库展示 ( export_final/export ) 👇", open=True):
                    gr.Markdown("> 浏览已导出的高质量 MP4 演示与 BVH 骨架序列。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            demo_bvh_dropdown = gr.Dropdown(choices=scan_export_final_bvh(), label="选择 BVH 动作序列", allow_custom_value=True)
                            demo_bvh_btn = gr.Button("🔍 使用 Blender 打开此 BVH", variant="secondary")
                        with gr.Column(scale=1):
                            demo_bvh_msg = gr.Markdown("准备就绪")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            demo_mp4_dropdown = gr.Dropdown(choices=scan_export_final_mp4(), label="选择 MP4 渲染视频", allow_custom_value=True)
                            demo_mp4_btn = gr.Button("▶️ 使用默认播放器打开", variant="primary")
                        with gr.Column(scale=1):
                            demo_mp4_msg = gr.Markdown("准备就绪")
                    
                    def load_demo_bvh(path):
                        if not path or not os.path.exists(path): return "❌ 文件不存在"
                        try:
                            from dunhuang_dance_gen.integrations import launch_blender_with_file
                            ok, msg = launch_blender_with_file(path)
                            return f"### ✅ {msg}" if ok else f"### ❌ {msg}"
                        except Exception as e:
                            return f"### ❌ 启动 Blender 失败: {e}"
                    
                    def load_demo_mp4(path):
                        if not path or not os.path.exists(path): return "❌ 文件不存在"
                        try:
                            import os
                            os.startfile(path)
                            return f"### ✅ 已尝试外部播放: {os.path.basename(path)}"
                        except Exception as e:
                            return f"### ❌ 打开视频失败: {e}"
                        
                    demo_bvh_btn.click(load_demo_bvh, inputs=[demo_bvh_dropdown], outputs=[demo_bvh_msg])
                    demo_mp4_btn.click(load_demo_mp4, inputs=[demo_mp4_dropdown], outputs=[demo_mp4_msg])

                with gr.Accordion("2. 进阶功能演示 · Inbetweening & Harmonization 👇", open=True):
                    gr.Markdown("""> 🎬 **进阶动作编辑能力演示** — 展示 SinMDM 在敦煌舞领域的 **动作补间 (Inbetweening)** 和 **动作风格和谐化 (Harmonization)** 成果。""")
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 🔗 动作补间 (Inbetweening)")
                            gr.Markdown("给定动作序列的起始与终止姿态，扩散模型自动生成自然过渡。")
                            adv_inbetween_video = gr.Video(
                                label="Inbetweening 演示 · 05-2-JiGuJiYue",
                                value=r"D:\sinMDM\export_final\video\save\05-2-JiGuJiYue\inbetweening_demo\sample00.mp4",
                                height=320,
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("#### 🎵 风格和谐化 (Harmonization)")
                            gr.Markdown("将外源行走动作和谐地融入击鼓击乐风格，保持节奏与姿态一致性。")
                            adv_harmonize_video = gr.Video(
                                label="Harmonization 演示 · Walking → JiGuJiYue",
                                value=r"D:\sinMDM\export_final\video\save\05-2-JiGuJiYue\harmonization_walking\sample01.mp4",
                                height=320,
                            )
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
                            ---
                            | 功能 | 技术原理 | 输入 | 输出 |
                            |------|---------|------|------|
                            | **Inbetweening** | 条件扩散 · 固定首尾帧 → 去噪扩散填充中间帧 | 起止关键姿态 | 平滑过渡动作序列 |
                            | **Harmonization** | 风格迁移 · 外源动作注入后，梯度引导匹配目标分布 | 外源动作 + 风格参考 | 和谐化动作序列 |
                            """)

                with gr.Accordion("3. 实时生成引擎 👇", open=False):
                    gr.Markdown("> 基于已有参数模型实时生成新动作。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            scan_models_btn = gr.Button("🩺 刷新并加载已训练的 SinMDM 参数", variant="secondary")
                            gen_model = gr.Dropdown(choices=get_available_models(), label="选择扩散模型 (Checkpoint)", allow_custom_value=True)
                            with gr.Row():
                                gen_samples = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="采样批次 (Samples)")
                                gen_length = gr.Slider(minimum=2.0, maximum=30.0, value=10.0, step=0.5, label="合成长度 (Seconds)")
                            with gr.Row():
                                gen_diversity = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="扰动/多样性 (Diversity)")
                                gen_seed = gr.Number(value=-1, label="全局种子 (-1=随机)", precision=0)
                                
                            gen_btn = gr.Button("✨ 生成骨骼动作", variant="primary")
                            model_scan_report = gr.Markdown()
                        
                        with gr.Column(scale=1):
                            gen_output = gr.Markdown("等待生成...")
                            gen_preview_file = gr.File(label="获取生成源文件 (BVH/MP4)")
                        with gr.Column(scale=1):
                            gen_video = gr.Video(label="生成效果前瞻 (MP4)", height=250)
                    
                    scan_models_btn.click(scan_saved_models_ui, outputs=[model_scan_report, gen_model])
                    gen_btn.click(generate_motion, inputs=[gen_model, gen_samples, gen_length, gen_seed, gen_diversity], outputs=[gen_output, gr.Textbox(visible=False), gen_video, gen_preview_file])
                
                with gr.Accordion("4. 生成结果 3D 对比 👇", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            vis_orig = gr.Dropdown(choices=bvh_files, label="参演角色 A (参考)", allow_custom_value=True)
                            vis_gen = gr.Dropdown(choices=bvh_files, label="参演角色 B (生成)", allow_custom_value=True)
                            vis_compare_frame = gr.Slider(minimum=0, maximum=500, value=0, step=1, label="全局同步时间轴")
                            vis_compare_btn = gr.Button("⚡ 同步投射对比", variant="primary")
                        with gr.Column(scale=2):
                            vis_compare_plot = gr.Plot(label="并行视场")
                    def compare_skeletons(orig, gen, frame):
                        return compare_bvh(orig, gen, int(frame)) if orig and gen else None
                    vis_compare_btn.click(compare_skeletons, inputs=[vis_orig, vis_gen, vis_compare_frame], outputs=[vis_compare_plot])

            # ==========================================
            # 模块二：【🎨 风格迁移与约束】
            # ==========================================
            with gr.TabItem("🎨 风格迁移与约束", id="style_engine"):
                gr.Markdown("### 🎨 风格引擎与精细控制 (基于流形或直接运动学限定)")
                gr.Markdown("> 直接将特色风格(如：飞天的 S 曲线、反弹琵琶的舒展度) 映射到目标动作上。建议选用 `export_final/export` 中的优质 BVH 作为风格源。")
                
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("**【模式 A】基于参考标本的风格转移**")
                        style_source_file = gr.File(label="📂 待处理目标动作 (Source)", file_types=[".bvh"])
                        style_ref_file = gr.File(label="📂 风格提供者动作 (Reference)", file_types=[".bvh"])
                        with gr.Row():
                            style_arm_slider = gr.Slider(minimum=0, maximum=1.0, value=0.5, step=0.1, label="🦾 上肢舒展力度")
                            style_spine_slider = gr.Slider(minimum=0, maximum=1.0, value=0.5, step=0.1, label="🐍 核心 S 形弯曲")
                            style_rhythm_slider = gr.Slider(minimum=0, maximum=1.0, value=0.3, step=0.1, label="🎵 顿挫/断点停顿")
                        with gr.Row():
                            style_amplitude_slider = gr.Slider(minimum=0, maximum=1.0, value=0.4, step=0.1, label="📊 上下半身发力比")
                            style_symmetry_slider = gr.Slider(minimum=0, maximum=1.0, value=0.3, step=0.1, label="⚖️ 镜像破除/对称感")
                            style_global_slider = gr.Slider(minimum=0, maximum=2.0, value=1.0, step=0.1, label="🎛️ 整体转移倍率")
                        style_transfer_btn = gr.Button("🔄 注入参考风格", variant="primary")

                    with gr.Column(scale=3):
                        gr.Markdown("**【模式 B】基于数值的几何强约束**")
                        constraint_source_file = gr.File(label="📂 待处理片段", file_types=[".bvh"])
                        with gr.Row():
                            c_arm = gr.Slider(minimum=50, maximum=200, value=110, step=5, label="目标展臂张角(°)")
                            c_spine = gr.Slider(minimum=50, maximum=400, value=250, step=10, label="目标脊柱曲度(°)")
                        with gr.Row():
                            c_ratio = gr.Slider(minimum=0.5, maximum=3.0, value=1.3, step=0.1, label="上下身挥动比")
                            c_pause = gr.Slider(minimum=0, maximum=60, value=12, step=1, label="静帧占用率(%)")
                        with gr.Row():
                            c_symmetry = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="刚体对称系数")
                            c_strength = gr.Slider(minimum=0, maximum=1.0, value=0.5, step=0.1, label="约束强制力")
                        constraint_btn = gr.Button("⚡ 应用几何限定", variant="secondary")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("**结果导出区**")
                        style_result_output = gr.Markdown("无活动")
                        style_download_file = gr.File(label="⬇️ 获取风格化结果 (BVH)")

                # 风格迁移的逻辑引用
                def run_style_transfer_ui(src, ref, arm, spine, rhythm, amp, sym, glob):
                    if src is None or ref is None: return "❌ 缺失 BVH 源", None
                    try:
                        from dunhuang_dance_gen.postprocess.style_transfer import DunhuangStyleTransfer, StyleTransferConfig
                        from dunhuang_dance_gen.export.bvh_writer import BVHWriter
                        import tempfile, copy
                        s = load_bvh(src.name)
                        r = load_bvh(ref.name)
                        cfg = StyleTransferConfig(arm, spine, rhythm, amp, sym, glob)
                        res = DunhuangStyleTransfer(fps=30.0).transfer(s.rotations, s.positions, r.rotations, cfg)
                        out_path = os.path.join(tempfile.gettempdir(), f"styled_{_timestamp()}.bvh")
                        out_data = copy.copy(s)
                        out_data.rotations = res.rotations
                        out_data.positions = res.positions
                        BVHWriter().write_from_bvhdata(out_path, out_data)
                        return f"""### ✅ 传递完成\n帧数: {res.rotations.shape[0]}, 对称性变迁: {res.source_style.overall_symmetry:.2f} -> {res.result_style.overall_symmetry:.2f}""", out_path
                    except Exception as e:
                        return f"❌ 错误: {e}", None

                def run_constraint_ui(src, a, s_p, r, p, s_y, s_t):
                    if src is None: return "❌ 缺失 BVH 源", None
                    try:
                        from dunhuang_dance_gen.postprocess.style_transfer import StyleConstraintApplicator
                        from dunhuang_dance_gen.export.bvh_writer import BVHWriter
                        import tempfile, copy
                        s = load_bvh(src.name)
                        mod_rot, mod_pos, rep = StyleConstraintApplicator(fps=30.0).apply(s.rotations, s.positions, a, s_p, r, p, s_y, s_t)
                        out_path = os.path.join(tempfile.gettempdir(), f"constrained_{_timestamp()}.bvh")
                        out_data = copy.copy(s)
                        out_data.rotations = mod_rot
                        out_data.positions = mod_pos
                        BVHWriter().write_from_bvhdata(out_path, out_data)
                        return "### ✅ 限定应用", out_path
                    except Exception as e:
                        return f"❌ 错误: {e}", None

                style_transfer_btn.click(run_style_transfer_ui, inputs=[style_source_file, style_ref_file, style_arm_slider, style_spine_slider, style_rhythm_slider, style_amplitude_slider, style_symmetry_slider, style_global_slider], outputs=[style_result_output, style_download_file])
                constraint_btn.click(run_constraint_ui, inputs=[constraint_source_file, c_arm, c_spine, c_ratio, c_pause, c_symmetry, c_strength], outputs=[style_result_output, style_download_file])



            # ==========================================
            # 模块四：【🧠 模型训练与优化】
            # ==========================================
            with gr.TabItem("🧠 模型训练与优化", id="training_opt"):
                gr.Markdown("### 🧠 SinMDM 扩散模型训练流程控制与生成的动作平滑")
                
                with gr.Accordion("1. 核心网络训练 👇", open=True):
                    with gr.Row():
                        with gr.Column(scale=1):
                            train_bvh = gr.Dropdown(choices=get_available_bvh_for_training(), label="单序列训练数据", info="此架构在极少样本下即可进行针对性过拟合", allow_custom_value=True)
                            with gr.Row():
                                train_arch = gr.Radio(choices=["qna", "unet"], value="qna", label="网络主干", info="QnA: 注意力流 | UNet: 传统卷积")
                                train_lr_gamma = gr.Number(value=0.99998, label="策略: 学习率 Gamma", precision=5)
                            with gr.Row():
                                train_steps = gr.Slider(minimum=5000, maximum=50000, value=20000, step=1000, label="迭代步数 (Iterations)")
                                train_save_interval = gr.Slider(minimum=1000, maximum=10000, value=2500, step=500, label="检查点频率")
                            train_gen = gr.Checkbox(value=True, label="训练期间渲染预览动画")
                            train_btn = gr.Button("🚀 后台提交训练任务", variant="primary")
                        with gr.Column(scale=2):
                            train_output = gr.Markdown(label="作业调度结果")
                            train_cmd = gr.Textbox(label="后台实际执行指令 (WSL环境兼容)", interactive=False, lines=2)
                    
                    train_btn.click(start_training, inputs=[train_bvh, train_steps, train_save_interval, train_arch, train_lr_gamma, train_gen], outputs=[train_output, train_cmd])
                
                with gr.Accordion("2. 运动学后处理与优化 👇", open=False):
                    gr.Markdown("> 消除模型生成特有的高频微抖、地平线穿模以及不符合解剖学的速度突变。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            export_bvh_dropdown = gr.Dropdown(choices=scan_export_final_bvh(), label="选择待修补的生成资产", allow_custom_value=True)
                            with gr.Row():
                                pp_smooth_method = gr.Radio(choices=["savgol", "gaussian", "none"], value="savgol", label="降噪算法")
                                pp_smooth_window = gr.Slider(minimum=3, maximum=15, value=5, step=2, label="核大小")
                            with gr.Row():
                                pp_fix_spikes = gr.Checkbox(value=True, label="极值速度截断修复")
                                pp_joint_limits = gr.Checkbox(value=True, label="人体运动学角位限制")
                                pp_stabilize = gr.Checkbox(value=True, label="重心锚定稳态约束")
                            pp_btn = gr.Button("⚡ 执行管线清洗并打包", variant="primary")
                        with gr.Column(scale=2):
                            pp_report = gr.Markdown()
                            pp_download = gr.File(label="获取纯净 BVH (Cleaned)")
                    pp_btn.click(apply_postprocess, inputs=[export_bvh_dropdown, pp_smooth_method, pp_smooth_window, pp_fix_spikes, pp_joint_limits, pp_stabilize], outputs=[pp_report, pp_download])



        gr.Markdown("""
        ---
        <center>
        
        **基于姿态序列的敦煌舞蹈动作生成系统** · SinMDM (ICLR 2024)  
        西北民族大学 · 数学与计算机科学学院 · 2022级计算机科学与技术  
        
        </center>
        """)
    
    return app


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
