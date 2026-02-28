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
import subprocess
import tempfile
import numpy as np
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

from dunhuang_dance_gen.data.bvh_parser import BVHParser, load_bvh
from dunhuang_dance_gen.postprocess import MotionSmoother, PhysicalConstraints, PostProcessPipeline, PostProcessConfig
from dunhuang_dance_gen.export.bvh_writer import BVHWriter
from dunhuang_dance_gen.evaluate.metrics import MotionEvaluator

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
    for root, dirs, filenames in os.walk(DATASET_DIR):
        for f in filenames:
            if f.endswith('.bvh'):
                files.append(os.path.join(root, f))
    
    # 也扫描 save 目录中的生成文件
    for root, dirs, filenames in os.walk(SAVE_DIR):
        for f in filenames:
            if f.endswith('.bvh'):
                files.append(os.path.join(root, f))
    
    return files


# ============================================================
# Tab 2: 模型训练
# ============================================================
def get_available_bvh_for_training():
    """获取可用于训练的 BVH 文件列表"""
    files = []
    for root, dirs, filenames in os.walk(DATASET_DIR):
        for f in filenames:
            if f.endswith('.bvh'):
                files.append(os.path.join(root, f))
    return files


def build_train_command(
    bvh_path: str,
    num_steps: int,
    save_interval: int,
    arch: str,
    lr_gamma: float,
    gen_during_training: bool,
):
    """构建训练命令"""
    cmd = [
        "python", "-m", "train.train_sinmdm",
        "--arch", arch,
        "--dataset", "bvh_general",
        "--sin_path", bvh_path,
        "--lr_method", "ExponentialLR",
        "--lr_gamma", str(lr_gamma),
        "--num_steps", str(num_steps),
        "--save_interval", str(save_interval),
        "--use_scale_shift_norm",
        "--use_checkpoint",
    ]
    if gen_during_training:
        cmd.append("--gen_during_training")
    cmd.append("--overwrite")
    
    return " ".join(cmd)


def start_training(bvh_path, num_steps, save_interval, arch, lr_gamma, gen_during):
    """启动训练（返回命令供用户执行）"""
    if not bvh_path:
        return "❌ 请先选择训练数据", ""
    
    cmd = build_train_command(bvh_path, int(num_steps), int(save_interval), 
                               arch, float(lr_gamma), gen_during)
    
    info = f"""## 🚀 训练命令已生成

```bash
{cmd}
```

### 训练配置
| 参数 | 值 |
|------|-----|
| **数据路径** | `{Path(bvh_path).name}` |
| **架构** | {arch} |
| **训练步数** | {num_steps} |
| **保存间隔** | {save_interval} |
| **学习率衰减** | {lr_gamma} |
| **训练中生成** | {'✅' if gen_during else '❌'} |

> ⚠️ 请在终端中执行上述命令。训练需要 GPU 环境。
> 建议在 AutoDL 等云 GPU 平台执行。
"""
    return info, cmd


# ============================================================
# Tab 3: 生成预览
# ============================================================
def get_available_models():
    """获取已训练模型列表"""
    models = []
    save_path = Path(SAVE_DIR)
    if not save_path.exists():
        return models
    for d in save_path.iterdir():
        if d.is_dir():
            pts = list(d.glob("model*.pt"))
            for pt in sorted(pts):
                models.append(str(pt))
    return models


def build_generate_command(model_path, num_samples, motion_length, seed):
    """构建生成命令"""
    cmd = [
        "python", "-m", "sample.generate",
        "--model_path", model_path,
        "--num_samples", str(int(num_samples)),
        "--motion_length", str(float(motion_length)),
    ]
    if seed >= 0:
        cmd.append("--seed")
        cmd.append(str(int(seed)))
    
    return " ".join(cmd)


def generate_motion(model_path, num_samples, motion_length, seed):
    """生成动作（返回命令和预览信息）"""
    if not model_path:
        return "❌ 请选择模型", "", None
    
    cmd = build_generate_command(model_path, num_samples, motion_length, seed)
    
    # 检查模型目录下是否已有生成结果
    model_dir = Path(model_path).parent
    existing_bvh = sorted(model_dir.glob("sample*.bvh"))
    existing_mp4 = sorted(model_dir.glob("*.mp4"))
    
    preview_info = f"""## 🎭 生成命令

```bash
{cmd}
```

### 参数设置
| 参数 | 值 |
|------|-----|
| **模型** | `{Path(model_path).name}` |
| **样本数** | {int(num_samples)} |
| **时长** | {motion_length} 秒 |
| **种子** | {'随机' if seed < 0 else int(seed)} |
"""
    
    if existing_bvh:
        preview_info += "\n### 已有生成结果\n"
        for f in existing_bvh[:5]:
            preview_info += f"- 📄 `{f.name}`\n"
    
    # 尝试找已有的 MP4 预览
    video_path = None
    if existing_mp4:
        video_path = str(existing_mp4[0])
    
    return preview_info, cmd, video_path


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
                        train_btn = gr.Button("🚀 生成训练命令", variant="primary")
                    
                    with gr.Column(scale=2):
                        train_output = gr.Markdown(label="训练信息")
                        train_cmd = gr.Textbox(
                            label="训练命令 (复制到终端执行)",
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
                        gen_model = gr.Dropdown(
                            choices=model_files,
                            label="选择模型",
                            info="已训练的模型检查点",
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
                        gen_output = gr.Markdown(label="生成信息")
                        gen_cmd = gr.Textbox(
                            label="生成命令",
                            interactive=True,
                            lines=2,
                        )
                        gen_video = gr.Video(label="动画预览", height=400)
                
                gen_btn.click(
                    generate_motion,
                    inputs=[gen_model, gen_samples, gen_length, gen_seed],
                    outputs=[gen_output, gen_cmd, gen_video]
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
                        pose_btn = gr.Button("🦴 提取姿态", variant="primary")
                    
                    with gr.Column(scale=2):
                        pose_output = gr.Markdown()
                
                def extract_pose(video_file, method, fps):
                    if video_file is None:
                        return "❌ 请先上传视频文件"
                    
                    # 检查 MediaPipe 是否可用
                    try:
                        import mediapipe
                        mp_available = True
                    except ImportError:
                        mp_available = False
                    
                    video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
                    
                    info = f"""## 🎥 视频姿态估计

| 参数 | 值 |
|------|-----|
| **视频** | `{Path(video_path).name}` |
| **方法** | {method} |
| **帧率** | {fps} FPS |
| **MediaPipe** | {'✅ 已安装' if mp_available else '❌ 未安装'} |

"""
                    if not mp_available:
                        info += """### ⚠️ 安装依赖

```bash
pip install mediapipe opencv-python
```

安装后即可使用视频姿态估计功能。

### 技术说明

本系统的数据来源以**公开敦煌舞三维动作数据集**为主（已集成 16 段 BVH 数据），
视频姿态估计作为补充数据源接口，支持用户从自有视频中提取额外训练数据。
"""
                    else:
                        info += """### ✅ 环境就绪

MediaPipe 已安装，可以进行视频姿态估计。

> ⚠️ 视频姿态估计生成的是 2D/3D 关键点坐标，需要经过**骨架映射和格式转换**
> 才能转为可训练的 BVH 格式。当前系统以公开敦煌舞 BVH 数据集为主要数据源。
"""
                    return info
                
                pose_btn.click(extract_pose, inputs=[pose_video, pose_method, pose_fps], outputs=[pose_output])
        
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
