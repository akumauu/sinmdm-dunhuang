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
        """
    ) as app:
        
        gr.Markdown("""
        # 🎭 敦煌舞蹈动作生成系统
        ### 基于 SinMDM 的单序列扩散模型 · 西北民族大学本科毕业设计
        """, elem_classes="main-title")
        
        with gr.Tabs():
            # ---- Tab 1: 数据管理 ----
            with gr.TabItem("📁 数据管理", id="data"):
                with gr.Row():
                    with gr.Column(scale=1):
                        bvh_dropdown = gr.Dropdown(
                            choices=bvh_files,
                            label="选择 BVH 文件",
                            info="从数据集或生成结果中选择",
                            allow_custom_value=True,
                        )
                        bvh_upload = gr.File(
                            label="或上传 BVH 文件",
                            file_types=[".bvh"],
                        )
                        load_btn = gr.Button("📊 加载并分析", variant="primary")
                        status_text = gr.Textbox(label="状态", interactive=False)
                    
                    with gr.Column(scale=2):
                        info_output = gr.Markdown(label="文件信息")
                        preview_img = gr.Image(label="骨架预览", height=400)
                
                def on_upload(file):
                    if file:
                        return file.name
                    return ""
                
                bvh_upload.change(on_upload, inputs=[bvh_upload], outputs=[bvh_dropdown])
                load_btn.click(
                    load_bvh_info,
                    inputs=[bvh_dropdown],
                    outputs=[info_output, status_text, preview_img]
                )
            
            # ---- Tab 2: 模型训练 ----
            with gr.TabItem("🧠 模型训练", id="train"):
                with gr.Row():
                    with gr.Column(scale=1):
                        train_bvh = gr.Dropdown(
                            choices=get_available_bvh_for_training(),
                            label="训练数据 (BVH)",
                            info="选择用于训练的敦煌舞 BVH 文件",
                            allow_custom_value=True,
                        )
                        train_arch = gr.Radio(
                            choices=["qna", "unet"],
                            value="qna",
                            label="网络架构",
                            info="QnA: 局部注意力(推荐) | UNet: 标准卷积"
                        )
                        train_steps = gr.Slider(
                            minimum=5000, maximum=50000, value=20000, step=1000,
                            label="训练步数",
                            info="推荐 10000-20000 步"
                        )
                        train_save_interval = gr.Slider(
                            minimum=1000, maximum=10000, value=2500, step=500,
                            label="保存间隔"
                        )
                        train_lr_gamma = gr.Number(
                            value=0.99998,
                            label="学习率衰减 (gamma)",
                            precision=5
                        )
                        train_gen = gr.Checkbox(
                            value=True,
                            label="训练中生成预览"
                        )
                        train_btn = gr.Button("🚀 启动训练", variant="primary")
                    
                    with gr.Column(scale=2):
                        train_output = gr.Markdown(label="训练信息")
                        train_cmd = gr.Textbox(
                            label="训练命令 (本次实际执行)",
                            interactive=True,
                            lines=3,
                        )
                
                train_btn.click(
                    start_training,
                    inputs=[train_bvh, train_steps, train_save_interval, 
                            train_arch, train_lr_gamma, train_gen],
                    outputs=[train_output, train_cmd]
                )
            
            # ---- Tab 3: 生成预览 ----
            with gr.TabItem("🎬 生成预览", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        scan_models_btn = gr.Button("🩺 扫描现有模型", variant="secondary")
                        gen_model = gr.Dropdown(
                            choices=get_available_models(),
                            label="选择模型",
                            info="默认展示主线 bvh_general 的最新检查点",
                            allow_custom_value=True,
                        )
                        gen_samples = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="生成样本数"
                        )
                        gen_length = gr.Slider(
                            minimum=2.0, maximum=30.0, value=10.0, step=0.5,
                            label="生成时长 (秒)"
                        )
                        gen_seed = gr.Number(
                            value=-1,
                            label="随机种子 (-1=随机)",
                            precision=0
                        )
                        gen_diversity = gr.Slider(
                            minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                            label="多样性系数",
                            info=">1.0 增大生成多样性, <1.0 更接近原始动作"
                        )
                        gen_btn = gr.Button("🎭 生成动作", variant="primary")
                    
                    with gr.Column(scale=2):
                        model_scan_report = gr.Markdown(label="模型体检")
                        gen_output = gr.Markdown(label="生成信息")
                        gen_cmd = gr.Textbox(
                            label="生成命令 (本次实际执行)",
                            interactive=True,
                            lines=2,
                        )
                        gen_video = gr.Video(label="动画预览", height=400)
                        gen_preview_file = gr.File(label="预览文件 (GIF/MP4)")
                
                scan_models_btn.click(
                    scan_saved_models_ui,
                    outputs=[model_scan_report, gen_model]
                )
                gen_btn.click(
                    generate_motion,
                    inputs=[gen_model, gen_samples, gen_length, gen_seed, gen_diversity],
                    outputs=[gen_output, gen_cmd, gen_video, gen_preview_file]
                )
            
            # ---- Tab 4: 导出与后处理 ----
            with gr.TabItem("📤 后处理与导出", id="export"):
                with gr.Row():
                    with gr.Column(scale=1):
                        export_bvh = gr.Dropdown(
                            choices=bvh_files,
                            label="选择待处理的 BVH 文件",
                            allow_custom_value=True,
                        )
                        
                        gr.Markdown("### ⚙️ 后处理参数")
                        pp_smooth_method = gr.Radio(
                            choices=["savgol", "gaussian", "none"],
                            value="savgol",
                            label="平滑方法"
                        )
                        pp_smooth_window = gr.Slider(
                            minimum=3, maximum=15, value=5, step=2,
                            label="滤波窗口大小"
                        )
                        pp_fix_spikes = gr.Checkbox(
                            value=True, label="修正速度突变"
                        )
                        pp_joint_limits = gr.Checkbox(
                            value=True, label="关节角度限位"
                        )
                        pp_stabilize = gr.Checkbox(
                            value=True, label="根节点运动稳定"
                        )
                        pp_btn = gr.Button("⚡ 执行后处理并导出", variant="primary")
                    
                    with gr.Column(scale=2):
                        pp_report = gr.Markdown(label="处理报告")
                        pp_download = gr.File(label="下载处理后的 BVH 文件")
                
                pp_btn.click(
                    apply_postprocess,
                    inputs=[export_bvh, pp_smooth_method, pp_smooth_window,
                            pp_fix_spikes, pp_joint_limits, pp_stabilize],
                    outputs=[pp_report, pp_download]
                )
            
            # ---- Tab 5: 3D 可视化 ----
            with gr.TabItem("🎯 3D 可视化", id="vis3d"):
                gr.Markdown("### 交互式三维骨架查看器")
                with gr.Row():
                    with gr.Column(scale=1):
                        vis_bvh = gr.Dropdown(
                            choices=bvh_files,
                            label="选择 BVH 文件",
                            allow_custom_value=True,
                        )
                        vis_frame = gr.Slider(
                            minimum=0, maximum=500, value=0, step=1,
                            label="帧索引"
                        )
                        vis_btn = gr.Button("🔍 查看骨架", variant="primary")
                        
                        gr.Markdown("### 对比回放")
                        vis_orig = gr.Dropdown(
                            choices=bvh_files,
                            label="原始动作 BVH",
                            allow_custom_value=True,
                        )
                        vis_gen = gr.Dropdown(
                            choices=bvh_files,
                            label="生成动作 BVH",
                            allow_custom_value=True,
                        )
                        vis_compare_frame = gr.Slider(
                            minimum=0, maximum=500, value=0, step=1,
                            label="对比帧"
                        )
                        vis_compare_btn = gr.Button("⚡ 并排对比", variant="secondary")
                    
                    with gr.Column(scale=2):
                        vis_plot = gr.Plot(label="3D 骨架查看")
                        vis_compare_plot = gr.Plot(label="对比查看")
                
                def view_skeleton(bvh_path, frame_idx):
                    if not bvh_path or not os.path.exists(bvh_path):
                        return None
                    try:
                        return visualize_bvh(bvh_path, int(frame_idx))
                    except Exception as e:
                        return None
                
                def compare_skeletons(orig_path, gen_path, frame_idx):
                    if not orig_path or not gen_path:
                        return None
                    try:
                        return compare_bvh(orig_path, gen_path, int(frame_idx))
                    except Exception as e:
                        return None
                
                vis_btn.click(view_skeleton, inputs=[vis_bvh, vis_frame], outputs=[vis_plot])
                vis_compare_btn.click(
                    compare_skeletons, 
                    inputs=[vis_orig, vis_gen, vis_compare_frame], 
                    outputs=[vis_compare_plot]
                )
            
            # ---- Tab 6: 项目管理 ----
            with gr.TabItem("📂 项目管理", id="projects"):
                gr.Markdown("### 项目式任务管理")
                gr.Markdown("> 以项目为单位管理不同舞蹈片段的生成任务")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        proj_name = gr.Textbox(
                            label="项目名称",
                            placeholder="例: 飞天_动作生成"
                        )
                        proj_bvh = gr.Dropdown(
                            choices=bvh_files,
                            label="关联数据 (BVH)",
                            allow_custom_value=True,
                        )
                        proj_notes = gr.Textbox(
                            label="备注",
                            lines=3,
                            placeholder="项目说明..."
                        )
                        with gr.Row():
                            proj_new_btn = gr.Button("🆕 新建项目", variant="primary")
                            proj_save_btn = gr.Button("💾 保存进度")
                        
                        gr.Markdown("### 历史项目")
                        proj_list = gr.Dropdown(
                            label="打开项目",
                            choices=[],
                        )
                        proj_open_btn = gr.Button("📂 打开")
                    
                    with gr.Column(scale=2):
                        proj_info = gr.Markdown(label="项目信息")
                
                def new_project(name, bvh, notes):
                    if not name:
                        return "❌ 请输入项目名称"
                    proj_data = {
                        "name": name,
                        "bvh_path": bvh or "",
                        "notes": notes or "",
                        "created": str(np.datetime64('now')),
                        "status": "已创建",
                    }
                    proj_file = os.path.join(PROJECT_DIR, f"{name}.json")
                    with open(proj_file, 'w', encoding='utf-8') as f:
                        json.dump(proj_data, f, ensure_ascii=False, indent=2)
                    return f"✅ 项目 **{name}** 已创建\n\n📄 配置文件: `{proj_file}`"
                
                def save_project(name, bvh, notes):
                    if not name:
                        return "❌ 请输入项目名称"
                    proj_file = os.path.join(PROJECT_DIR, f"{name}.json")
                    proj_data = {}
                    if os.path.exists(proj_file):
                        with open(proj_file, 'r', encoding='utf-8') as f:
                            proj_data = json.load(f)
                    proj_data.update({
                        "name": name,
                        "bvh_path": bvh or proj_data.get("bvh_path", ""),
                        "notes": notes or proj_data.get("notes", ""),
                        "last_saved": str(np.datetime64('now')),
                        "status": "进行中",
                    })
                    with open(proj_file, 'w', encoding='utf-8') as f:
                        json.dump(proj_data, f, ensure_ascii=False, indent=2)
                    return f"✅ 项目 **{name}** 已保存"
                
                def list_projects():
                    projs = glob.glob(os.path.join(PROJECT_DIR, "*.json"))
                    return [Path(p).stem for p in projs]
                
                def open_project(name):
                    if not name:
                        return "❌ 请选择项目"
                    proj_file = os.path.join(PROJECT_DIR, f"{name}.json")
                    if not os.path.exists(proj_file):
                        return "❌ 项目文件不存在"
                    with open(proj_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    info = f"## 📂 项目: {data.get('name', name)}\n\n"
                    info += f"| 属性 | 值 |\n|---|---|\n"
                    for k, v in data.items():
                        info += f"| {k} | {v} |\n"
                    return info
                
                proj_new_btn.click(new_project, inputs=[proj_name, proj_bvh, proj_notes], outputs=[proj_info])
                proj_save_btn.click(save_project, inputs=[proj_name, proj_bvh, proj_notes], outputs=[proj_info])
                proj_open_btn.click(open_project, inputs=[proj_list], outputs=[proj_info])
                app.load(lambda: gr.update(choices=list_projects()), outputs=[proj_list])
            
            # ---- Tab 7: 视频姿态估计 ----
            with gr.TabItem("🎥 视频姿态估计", id="pose"):
                gr.Markdown("""### 视频到姿态序列提取
> 从敦煌舞视频中提取骨骼关键点序列，转化为可训练的 BVH 数据
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        pose_video = gr.File(
                            label="上传视频 (MP4/AVI)",
                            file_types=[".mp4", ".avi", ".mov"],
                        )
                        pose_method = gr.Radio(
                            choices=["MediaPipe", "OpenPose"],
                            value="MediaPipe",
                            label="姿态估计方法",
                            info="MediaPipe: 轻量/易部署 | OpenPose: 精度高"
                        )
                        pose_fps = gr.Slider(
                            minimum=15, maximum=60, value=30, step=5,
                            label="输出帧率"
                        )
                        pose_openpose_bin = gr.Textbox(
                            label="OpenPose 可执行文件 (可选)",
                            placeholder="留空则读取 OPENPOSE_BIN / OPENPOSE_ROOT",
                        )
                        pose_btn = gr.Button("🦴 提取姿态", variant="primary")
                    
                    with gr.Column(scale=2):
                        pose_output = gr.Markdown()
                        pose_download = gr.File(label="导出 BVH")
                
                def extract_pose(video_file, method, fps, openpose_bin):
                    if video_file is None:
                        return "❌ 请先上传视频文件", None

                    video_path = _coerce_local_path(video_file)
                    output_dir = Path(OUTPUT_DIR) / "pose_bvh"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{Path(video_path).stem}_{_timestamp()}.bvh"

                    try:
                        result = extract_video_to_bvh(
                            video_path=video_path,
                            output_path=str(output_path),
                            method=method,
                            target_fps=float(fps),
                            openpose_bin=(openpose_bin or "").strip() or None,
                        )
                    except Exception as e:
                        info = f"""## ❌ 姿态提取失败

| 参数 | 值 |
|------|-----|
| **视频** | `{Path(video_path).name}` |
| **方法** | {method} |
| **帧率** | {fps} FPS |

```text
{e}
```
"""
                        return info, None

                    notes = ""
                    if result.notes:
                        notes = "\n### 处理说明\n" + "\n".join([f"- {note}" for note in result.notes])

                    info = f"""## ✅ 视频姿态提取完成

| 参数 | 值 |
|------|-----|
| **视频** | `{Path(video_path).name}` |
| **请求方法** | {result.method_requested} |
| **实际方法** | {result.method_used} |
| **源帧率** | {result.source_fps:.2f} FPS |
| **输出帧率** | {result.output_fps:.2f} FPS |
| **有效帧数** | {result.frames_processed} |
| **补帧/丢帧数** | {result.dropped_frames} |
| **平均可见度** | {result.avg_visibility:.3f} |
| **动作时长** | {result.duration:.2f} 秒 |
| **输出 BVH** | `{result.output_bvh_path}` |

### 下一步

该 BVH 已可直接用于：
- 数据分析与 3D 预览
- 单序列 SinMDM 训练
- 后处理与风格迁移
{notes}
"""
                    return info, result.output_bvh_path
                
                pose_btn.click(
                    extract_pose,
                    inputs=[pose_video, pose_method, pose_fps, pose_openpose_bin],
                    outputs=[pose_output, pose_download],
                )

            # ---- Tab 8: 数据集构建 ----
            with gr.TabItem("🗂️ 数据集构建", id="dataset"):
                gr.Markdown("""### 训练/验证集组织与切片
> 将长时 BVH 动作切分为规范化 clip，并导出确定性的 train/val 列表
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_source_root = gr.Textbox(
                            value=DATASET_DIR,
                            label="BVH 数据根目录",
                            placeholder="例如: 敦煌舞三维动作数据集/长动作",
                        )
                        dataset_clip_seconds = gr.Slider(
                            minimum=2.0, maximum=12.0, value=4.0, step=0.5,
                            label="clip 时长 (秒)"
                        )
                        dataset_overlap_seconds = gr.Slider(
                            minimum=0.0, maximum=6.0, value=1.0, step=0.5,
                            label="clip 重叠 (秒)"
                        )
                        dataset_min_seconds = gr.Slider(
                            minimum=1.0, maximum=6.0, value=2.0, step=0.5,
                            label="最小保留时长 (秒)"
                        )
                        dataset_val_ratio = gr.Slider(
                            minimum=0.05, maximum=0.5, value=0.2, step=0.05,
                            label="验证集比例"
                        )
                        dataset_btn = gr.Button("📦 构建数据集", variant="primary")

                    with gr.Column(scale=2):
                        dataset_output = gr.Markdown()
                        dataset_package = gr.File(label="下载数据集包 (ZIP)")

                dataset_btn.click(
                    build_dataset_package,
                    inputs=[
                        dataset_source_root,
                        dataset_clip_seconds,
                        dataset_overlap_seconds,
                        dataset_min_seconds,
                        dataset_val_ratio,
                    ],
                    outputs=[dataset_output, dataset_package],
                )

            # ---- Tab 9: 教学分析与外部联动 ----
            with gr.TabItem("🎓 教学分析与联动", id="teaching"):
                gr.Markdown("""### 教学拆解 / 难度分级 / 外部专业工具联动
> 系统内完成拆解和导出，专业播放与精修可一键交给 Blender
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        teach_bvh = gr.Dropdown(
                            choices=scan_bvh_files(),
                            label="选择 BVH 文件",
                            allow_custom_value=True,
                        )
                        teach_segment_seconds = gr.Slider(
                            minimum=1.5, maximum=8.0, value=3.0, step=0.5,
                            label="目标分段时长 (秒)"
                        )
                        teach_min_seconds = gr.Slider(
                            minimum=1.0, maximum=4.0, value=1.5, step=0.5,
                            label="最小分段时长 (秒)"
                        )
                        teach_slow_factor = gr.Slider(
                            minimum=1.0, maximum=4.0, value=2.0, step=0.5,
                            label="慢放倍数"
                        )
                        teach_btn = gr.Button("🧭 生成教学包", variant="primary")

                        gr.Markdown("### Blender 联动")
                        blender_path = gr.Textbox(
                            label="Blender 可执行文件 (可选)",
                            placeholder="留空则自动搜索系统中的 Blender",
                        )
                        blender_btn = gr.Button("🧱 用 Blender 打开 BVH", variant="secondary")

                    with gr.Column(scale=2):
                        teach_output = gr.Markdown()
                        teach_package = gr.File(label="教学包 (ZIP)")
                        teach_slow_bvh = gr.File(label="慢放 BVH")
                        blender_output = gr.Markdown()

                teach_btn.click(
                    generate_teaching_package,
                    inputs=[teach_bvh, teach_segment_seconds, teach_min_seconds, teach_slow_factor],
                    outputs=[teach_output, teach_package, teach_slow_bvh],
                )
                blender_btn.click(
                    open_bvh_in_blender,
                    inputs=[teach_slow_bvh, teach_bvh, blender_path],
                    outputs=[blender_output],
                )
            
            # ---- 风格迁移与约束 ----
            with gr.Accordion("🎨 风格迁移与约束", open=True):
                gr.Markdown("""### 敦煌舞风格迁移与可控约束
                
将参考动作的风格特征(如上肢舒展度、脊柱 S 曲线)迁移到生成动作上,
或通过滑块直接控制目标风格参数。输出为**修改后的 BVH 文件**。
""")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 输入选择")
                        
                        # BVH 文件选择
                        style_source_file = gr.File(
                            label="📂 源动作 BVH (要修改的)",
                            file_types=[".bvh"]
                        )
                        style_ref_file = gr.File(
                            label="📂 参考动作 BVH (风格来源)",
                            file_types=[".bvh"]
                        )
                        
                        gr.Markdown("#### 风格迁移强度")
                        style_arm_slider = gr.Slider(
                            minimum=0, maximum=1.0, value=0.5, step=0.1,
                            label="🦾 上肢舒展度迁移"
                        )
                        style_spine_slider = gr.Slider(
                            minimum=0, maximum=1.0, value=0.5, step=0.1,
                            label="🐍 脊柱 S 曲线迁移"
                        )
                        style_rhythm_slider = gr.Slider(
                            minimum=0, maximum=1.0, value=0.3, step=0.1,
                            label="🎵 节奏停顿迁移"
                        )
                        style_amplitude_slider = gr.Slider(
                            minimum=0, maximum=1.0, value=0.4, step=0.1,
                            label="📊 动作幅度迁移"
                        )
                        style_symmetry_slider = gr.Slider(
                            minimum=0, maximum=1.0, value=0.3, step=0.1,
                            label="⚖️ 左右对称性迁移"
                        )
                        style_global_slider = gr.Slider(
                            minimum=0, maximum=2.0, value=1.0, step=0.1,
                            label="🎛️ 全局强度系数"
                        )
                        
                        style_transfer_btn = gr.Button("🔄 执行风格迁移", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 直接风格约束 (无需参考动作)")
                        
                        constraint_source_file = gr.File(
                            label="📂 待约束 BVH 文件",
                            file_types=[".bvh"]
                        )
                        
                        c_arm = gr.Slider(
                            minimum=50, maximum=200, value=110, step=5,
                            label="目标上肢舒展度 (°)"
                        )
                        c_spine = gr.Slider(
                            minimum=50, maximum=400, value=250, step=10,
                            label="目标脊柱弯曲度 (°)"
                        )
                        c_ratio = gr.Slider(
                            minimum=0.5, maximum=3.0, value=1.3, step=0.1,
                            label="目标上下身幅度比"
                        )
                        c_pause = gr.Slider(
                            minimum=0, maximum=60, value=12, step=1,
                            label="目标停顿占比 (%)"
                        )
                        c_symmetry = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.6, step=0.05,
                            label="目标整体对称性 (0~1)"
                        )
                        c_strength = gr.Slider(
                            minimum=0, maximum=1.0, value=0.5, step=0.1,
                            label="🎛️ 约束强度"
                        )
                        
                        constraint_btn = gr.Button("⚡ 应用风格约束", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 风格混合 (两段动作线性混合)")
                        blend_a_file = gr.File(
                            label="📂 动作 A BVH",
                            file_types=[".bvh"]
                        )
                        blend_b_file = gr.File(
                            label="📂 动作 B BVH",
                            file_types=[".bvh"]
                        )
                        blend_weight = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="混合权重 (0=全A, 1=全B)"
                        )
                        blend_btn = gr.Button("🧪 执行风格混合", variant="secondary")
                
                with gr.Row():
                    style_result_output = gr.Markdown()
                    style_download_file = gr.File(label="⬇️ 下载修改后的 BVH")
                
                def run_style_transfer(source_file, ref_file, arm_s, spine_s, rhythm_s, amp_s, symmetry_s, global_s):
                    if source_file is None or ref_file is None:
                        return "❌ 请上传源动作和参考动作 BVH 文件", None
                    
                    try:
                        from dunhuang_dance_gen.postprocess.style_transfer import (
                            DunhuangStyleTransfer, StyleTransferConfig
                        )
                        from dunhuang_dance_gen.export.bvh_writer import BVHWriter
                        
                        src = load_bvh(source_file.name if hasattr(source_file, 'name') else str(source_file))
                        ref = load_bvh(ref_file.name if hasattr(ref_file, 'name') else str(ref_file))
                        
                        cfg = StyleTransferConfig(
                            arm_extension_strength=arm_s,
                            spine_curvature_strength=spine_s,
                            rhythm_strength=rhythm_s,
                            amplitude_strength=amp_s,
                            symmetry_strength=symmetry_s,
                            global_strength=global_s,
                        )
                        
                        transfer = DunhuangStyleTransfer(fps=30.0)
                        result = transfer.transfer(
                            src.rotations, src.positions, ref.rotations, cfg
                        )
                        
                        # 导出修改后的 BVH
                        import tempfile, copy
                        out_path = os.path.join(tempfile.gettempdir(), "style_transferred.bvh")
                        writer = BVHWriter()
                        out_data = copy.copy(src)
                        out_data.rotations = result.rotations
                        out_data.positions = result.positions
                        writer.write_from_bvhdata(out_path, out_data)
                        
                        # 报告
                        report = f"""### ✅ 风格迁移完成

| 指标 | 源动作 | 参考动作 | 迁移结果 |
|------|--------|---------|---------|
| 上肢舒展度(°) | {result.source_style.left_arm_extension_mean:.1f} | {result.target_style.left_arm_extension_mean:.1f} | {result.result_style.left_arm_extension_mean:.1f} |
| 脊柱弯曲度(°) | {result.source_style.spine_curvature_mean:.1f} | {result.target_style.spine_curvature_mean:.1f} | {result.result_style.spine_curvature_mean:.1f} |
| 停顿占比(%) | {result.source_style.pause_ratio:.1f} | {result.target_style.pause_ratio:.1f} | {result.result_style.pause_ratio:.1f} |
| 上下身幅度比 | {result.source_style.upper_lower_ratio:.2f} | {result.target_style.upper_lower_ratio:.2f} | {result.result_style.upper_lower_ratio:.2f} |
| 整体对称性 | {result.source_style.overall_symmetry:.3f} | {result.target_style.overall_symmetry:.3f} | {result.result_style.overall_symmetry:.3f} |

输出帧数: {result.rotations.shape[0]}, 关节数: {result.rotations.shape[1]}
"""
                        return report, out_path
                    except Exception as e:
                        return f"❌ 风格迁移失败: {str(e)}", None
                
                def run_constraint(source_file, arm_target, spine_target, ratio_target, pause_target, symmetry_target, strength):
                    if source_file is None:
                        return "❌ 请上传 BVH 文件", None
                    
                    try:
                        from dunhuang_dance_gen.postprocess.style_transfer import StyleConstraintApplicator
                        from dunhuang_dance_gen.export.bvh_writer import BVHWriter
                        
                        src = load_bvh(source_file.name if hasattr(source_file, 'name') else str(source_file))
                        
                        applicator = StyleConstraintApplicator(fps=30.0)
                        mod_rot, mod_pos, change_report = applicator.apply(
                            src.rotations, src.positions,
                            target_arm_extension=arm_target,
                            target_spine_curvature=spine_target,
                            target_upper_lower_ratio=ratio_target,
                            target_pause_ratio=pause_target,
                            target_symmetry=symmetry_target,
                            constraint_strength=strength,
                        )
                        
                        import tempfile, copy
                        out_path = os.path.join(tempfile.gettempdir(), "style_constrained.bvh")
                        writer = BVHWriter()
                        out_data = copy.copy(src)
                        out_data.rotations = mod_rot
                        out_data.positions = mod_pos
                        writer.write_from_bvhdata(out_path, out_data)
                        
                        # 往返验证
                        reloaded = load_bvh(out_path)
                        
                        report = "### ✅ 风格约束已应用\n\n"
                        report += "| 维度 | 变化 |\n|------|------|\n"
                        for k, v in change_report.items():
                            report += f"| {k} | {v} |\n"
                        report += f"\n输出帧数: {mod_rot.shape[0]}, 关节数: {mod_rot.shape[1]}"
                        report += f"\n\n✅ BVH 往返验证通过 (重加载 {reloaded.num_frames} 帧, {reloaded.num_joints} 关节)"
                        
                        return report, out_path
                    except Exception as e:
                        return f"❌ 约束应用失败: {str(e)}", None

                def run_style_blend(file_a, file_b, weight):
                    if file_a is None or file_b is None:
                        return "❌ 请上传动作 A 和动作 B 的 BVH 文件", None

                    try:
                        import copy

                        src_a = load_bvh(_coerce_local_path(file_a))
                        src_b = load_bvh(_coerce_local_path(file_b))
                        mixed = style_blend(src_a.rotations, src_b.rotations, float(weight))

                        out_data = copy.copy(src_a)
                        out_data.rotations = mixed
                        out_data.positions = src_a.positions[: mixed.shape[0]].copy()
                        out_data.num_frames = mixed.shape[0]

                        out_path = os.path.join(tempfile.gettempdir(), "style_blended.bvh")
                        BVHWriter().write_from_bvhdata(out_path, out_data)

                        report = f"""### ✅ 风格混合完成

| 参数 | 值 |
|------|----|
| 动作 A | `{Path(_coerce_local_path(file_a)).name}` |
| 动作 B | `{Path(_coerce_local_path(file_b)).name}` |
| 混合权重 | {float(weight):.2f} |
| 输出帧数 | {mixed.shape[0]} |
| 关节数 | {mixed.shape[1]} |
"""
                        return report, out_path
                    except Exception as e:
                        return f"❌ 风格混合失败: {str(e)}", None
                
                style_transfer_btn.click(
                    run_style_transfer,
                    inputs=[style_source_file, style_ref_file, 
                            style_arm_slider, style_spine_slider,
                            style_rhythm_slider, style_amplitude_slider,
                            style_symmetry_slider, style_global_slider],
                    outputs=[style_result_output, style_download_file]
                )
                constraint_btn.click(
                    run_constraint,
                    inputs=[constraint_source_file, c_arm, c_spine, c_ratio, c_pause, c_symmetry, c_strength],
                    outputs=[style_result_output, style_download_file]
                )
                blend_btn.click(
                    run_style_blend,
                    inputs=[blend_a_file, blend_b_file, blend_weight],
                    outputs=[style_result_output, style_download_file]
                )
        
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
