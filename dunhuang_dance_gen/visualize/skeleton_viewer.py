"""
3D 骨骼可视化模块 — 基于 Plotly
支持:
    - 单帧 3D 骨架交互查看 (旋转/缩放/平移)
    - 动画帧序列播放
    - 原始 vs 生成动作并排对比
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple


# 敦煌舞 22 关节骨骼连接定义
# (parent_idx, child_idx)
DUNHUANG_SKELETON = [
    (0, 1), (1, 2), (2, 3),       # 脊柱: Hips → Spine → Chest → Neck
    (3, 4),                         # 头: Neck → Head
    (3, 5), (5, 6), (6, 7),       # 左臂: Chest → L_Shoulder → L_Elbow → L_Wrist
    (3, 8), (8, 9), (9, 10),      # 右臂: Chest → R_Shoulder → R_Elbow → R_Wrist
    (0, 11), (11, 12), (12, 13),  # 左腿: Hips → L_Hip → L_Knee → L_Ankle
    (0, 14), (14, 15), (15, 16),  # 右腿: Hips → R_Hip → R_Knee → R_Ankle
    (13, 17),                       # 左脚: L_Ankle → L_Foot
    (16, 18),                       # 右脚: R_Ankle → R_Foot
    (7, 19),                        # 左手: L_Wrist → L_Hand
    (10, 20),                       # 右手: R_Wrist → R_Hand
    (4, 21),                        # 头顶: Head → HeadTop
]

# 关节颜色分组
JOINT_COLORS = {
    'spine': '#4FC3F7',    # 蓝
    'left_arm': '#81C784', # 绿
    'right_arm': '#FF8A65', # 橙
    'left_leg': '#CE93D8', # 紫
    'right_leg': '#FFD54F', # 黄
    'extremity': '#E0E0E0', # 灰
}


def _positions_from_rotations(rotations: np.ndarray, 
                               positions: np.ndarray,
                               joint_names: List[str]) -> np.ndarray:
    """
    从旋转和根位置生成简易 3D 坐标
    
    对于 BVH 数据，使用旋转的简化前向运动学:
    将每个关节的旋转幅度映射为相对偏移
    """
    T = rotations.shape[0]
    n_joints = rotations.shape[1] if rotations.ndim == 3 else len(joint_names)
    
    # 使用固定骨骼长度的简化骨架
    bone_lengths = np.ones(n_joints) * 5.0  # 默认骨长

    # 简化: 使用旋转数据的前3个分量作为局部偏移
    coords = np.zeros((T, n_joints, 3))
    
    if rotations.ndim == 3:
        rot_data = rotations
    else:
        rot_data = rotations.reshape(T, -1, 3) if rotations.shape[1] % 3 == 0 else rotations
    
    # 为每个关节计算简化位置
    for t in range(T):
        # 根节点使用位置数据
        if positions is not None and len(positions) > t:
            if positions.ndim == 2 and positions.shape[1] >= 3:
                coords[t, 0] = positions[t, :3]
            elif positions.ndim == 1:
                coords[t, 0] = positions[:3]
        
        # 其他关节使用旋转驱动
        if rot_data.ndim == 3 and rot_data.shape[1] >= n_joints:
            for bone in DUNHUANG_SKELETON:
                if bone[0] < n_joints and bone[1] < n_joints:
                    parent = bone[0]
                    child = bone[1]
                    
                    # 使用旋转角度生成偏移方向
                    angle_rad = np.deg2rad(rot_data[t, child])
                    
                    # 简化 FK: 沿默认方向 + 旋转偏移
                    default_dir = np.array([0, 1, 0])  # 默认向上
                    
                    if child in [11, 12, 13, 14, 15, 16, 17, 18]:  # 腿部向下
                        default_dir = np.array([0, -1, 0])
                    elif child in [5, 6, 7, 19]:  # 左臂向左
                        default_dir = np.array([-1, 0, 0])
                    elif child in [8, 9, 10, 20]:  # 右臂向右
                        default_dir = np.array([1, 0, 0])
                    
                    offset = default_dir * bone_lengths[child]
                    coords[t, child] = coords[t, parent] + offset
    
    return coords


def render_skeleton_frame(positions_3d: np.ndarray, 
                           frame_idx: int = 0,
                           title: str = "骨架预览",
                           width: int = 600, 
                           height: int = 500) -> go.Figure:
    """
    渲染单帧 3D 骨架
    
    Args:
        positions_3d: (T, J, 3) 或 (J, 3) 关节位置
        frame_idx: 帧索引
    """
    if positions_3d.ndim == 3:
        pos = positions_3d[frame_idx]
    else:
        pos = positions_3d
    
    n_joints = pos.shape[0]
    
    fig = go.Figure()
    
    # 绘制骨骼连接线
    for parent, child in DUNHUANG_SKELETON:
        if parent < n_joints and child < n_joints:
            fig.add_trace(go.Scatter3d(
                x=[pos[parent, 0], pos[child, 0]],
                y=[pos[parent, 2], pos[child, 2]],  # Z as depth
                z=[pos[parent, 1], pos[child, 1]],  # Y as up
                mode='lines',
                line=dict(color='#4FC3F7', width=4),
                showlegend=False,
                hoverinfo='skip',
            ))
    
    # 绘制关节点
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 2],
        z=pos[:, 1],
        mode='markers',
        marker=dict(size=5, color='#FF8A65', symbol='circle'),
        text=[f'J{i}' for i in range(n_joints)],
        hoverinfo='text',
        showlegend=False,
    ))
    
    # 布局
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Z (深度)',
            zaxis_title='Y (高度)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    return fig


def render_comparison(pos_original: np.ndarray,
                       pos_generated: np.ndarray,
                       frame_idx: int = 0,
                       title_left: str = "原始动作",
                       title_right: str = "生成动作",
                       width: int = 1000,
                       height: int = 500) -> go.Figure:
    """
    并排对比两个骨架
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=[title_left, title_right],
        horizontal_spacing=0.05,
    )
    
    for col_idx, (pos_3d, color) in enumerate([
        (pos_original, '#4FC3F7'),
        (pos_generated, '#FF8A65'),
    ], 1):
        if pos_3d.ndim == 3:
            pos = pos_3d[min(frame_idx, len(pos_3d)-1)]
        else:
            pos = pos_3d
        
        n_joints = pos.shape[0]
        scene_name = f'scene{col_idx}' if col_idx > 1 else 'scene'
        
        for parent, child in DUNHUANG_SKELETON:
            if parent < n_joints and child < n_joints:
                fig.add_trace(go.Scatter3d(
                    x=[pos[parent, 0], pos[child, 0]],
                    y=[pos[parent, 2], pos[child, 2]],
                    z=[pos[parent, 1], pos[child, 1]],
                    mode='lines',
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo='skip',
                ), row=1, col=col_idx)
        
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 2], z=pos[:, 1],
            mode='markers',
            marker=dict(size=4, color=color),
            showlegend=False,
        ), row=1, col=col_idx)
    
    camera = dict(eye=dict(x=1.5, y=1.5, z=0.5))
    fig.update_layout(
        width=width, height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode='data', camera=camera,
                   xaxis_title='X', yaxis_title='Z', zaxis_title='Y'),
        scene2=dict(aspectmode='data', camera=camera,
                    xaxis_title='X', yaxis_title='Z', zaxis_title='Y'),
    )
    
    return fig


def render_animation_html(positions_3d: np.ndarray,
                           fps: int = 30,
                           title: str = "动作回放",
                           max_frames: int = 200) -> str:
    """
    生成可播放的 3D 动画 HTML
    使用 Plotly animation frames
    """
    T = min(positions_3d.shape[0], max_frames)
    n_joints = positions_3d.shape[1]
    step = max(1, positions_3d.shape[0] // max_frames)
    
    # 初始帧
    pos = positions_3d[0]
    
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": [],
    }
    
    # 骨骼线 + 关节点
    bone_x, bone_y, bone_z = [], [], []
    for parent, child in DUNHUANG_SKELETON:
        if parent < n_joints and child < n_joints:
            bone_x.extend([pos[parent, 0], pos[child, 0], None])
            bone_y.extend([pos[parent, 2], pos[child, 2], None])
            bone_z.extend([pos[parent, 1], pos[child, 1], None])
    
    fig_dict["data"].append({
        "type": "scatter3d",
        "x": bone_x, "y": bone_y, "z": bone_z,
        "mode": "lines",
        "line": {"color": "#4FC3F7", "width": 4},
        "hoverinfo": "skip",
    })
    
    fig_dict["data"].append({
        "type": "scatter3d",
        "x": pos[:, 0].tolist(),
        "y": pos[:, 2].tolist(),
        "z": pos[:, 1].tolist(),
        "mode": "markers",
        "marker": {"size": 5, "color": "#FF8A65"},
    })
    
    # 动画帧
    for i in range(0, positions_3d.shape[0], step):
        pos = positions_3d[i]
        bone_x, bone_y, bone_z = [], [], []
        for parent, child in DUNHUANG_SKELETON:
            if parent < n_joints and child < n_joints:
                bone_x.extend([pos[parent, 0], pos[child, 0], None])
                bone_y.extend([pos[parent, 2], pos[child, 2], None])
                bone_z.extend([pos[parent, 1], pos[child, 1], None])
        
        frame = {
            "data": [
                {"x": bone_x, "y": bone_y, "z": bone_z},
                {"x": pos[:, 0].tolist(), "y": pos[:, 2].tolist(), "z": pos[:, 1].tolist()},
            ],
            "name": f"frame_{i}",
        }
        fig_dict["frames"].append(frame)
    
    # 播放控制
    fig_dict["layout"] = {
        "title": {"text": title},
        "scene": {
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 0.5}},
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Z"},
            "zaxis": {"title": "Y (高度)"},
        },
        "updatemenus": [{
            "type": "buttons",
            "showactive": False,
            "y": 0,
            "x": 0.5,
            "xanchor": "center",
            "buttons": [
                {"label": "▶ 播放", "method": "animate",
                 "args": [None, {"frame": {"duration": int(1000/fps), "redraw": True},
                                  "fromcurrent": True}]},
                {"label": "⏸ 暂停", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
            ],
        }],
        "sliders": [{
            "active": 0,
            "steps": [
                {"args": [[f"frame_{i}"],
                          {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                 "label": f"{i}",
                 "method": "animate"}
                for i in range(0, positions_3d.shape[0], step)
            ],
            "x": 0.1, "len": 0.8,
            "currentvalue": {"prefix": "帧: ", "visible": True},
        }],
        "width": 800,
        "height": 600,
    }
    
    import plotly.io as pio
    fig = go.Figure(fig_dict)
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')


def visualize_bvh(bvh_path: str, frame_idx: int = 0) -> go.Figure:
    """
    从 BVH 文件直接生成 3D 可视化
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    
    data = load_bvh(bvh_path)
    positions_3d = _positions_from_rotations(data.rotations, data.positions, data.joint_names)
    
    return render_skeleton_frame(
        positions_3d, frame_idx,
        title=f"{Path(bvh_path).stem} (帧 {frame_idx}/{data.num_frames})"
    )


def compare_bvh(original_path: str, generated_path: str, frame_idx: int = 0) -> go.Figure:
    """
    对比两个 BVH 文件
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dunhuang_dance_gen.data.bvh_parser import load_bvh
    
    orig = load_bvh(original_path)
    gen = load_bvh(generated_path)
    
    orig_pos = _positions_from_rotations(orig.rotations, orig.positions, orig.joint_names)
    gen_pos = _positions_from_rotations(gen.rotations, gen.positions, gen.joint_names)
    
    return render_comparison(
        orig_pos, gen_pos, frame_idx,
        title_left=f"原始: {Path(original_path).stem}",
        title_right=f"生成: {Path(generated_path).stem}",
    )
