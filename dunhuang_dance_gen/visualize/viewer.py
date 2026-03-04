"""
Motion Viewer for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 3D动作可视化模块

提供简单的 3D 骨骼可视化功能
"""

import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path


class MotionViewer:
    """3D 动作可视化器"""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 10),
        fps: float = 30.0,
        elev: float = 15,
        azim: float = 45
    ):
        """
        初始化可视化器
        
        Args:
            figsize: 图像尺寸
            fps: 播放帧率
            elev: 视角仰角
            azim: 视角方位角
        """
        self.figsize = figsize
        self.fps = fps
        self.elev = elev
        self.azim = azim
        
        self.fig = None
        self.ax = None
        self.animation = None
    
    def plot_skeleton(
        self,
        joint_positions: np.ndarray,
        parent_indices: np.ndarray,
        joint_names: Optional[List[str]] = None,
        show_labels: bool = False,
        title: str = "Skeleton"
    ):
        """
        绘制单帧骨架
        
        Args:
            joint_positions: 关节位置 (joints, 3)
            parent_indices: 父节点索引
            joint_names: 关节名称（可选）
            show_labels: 是否显示关节标签
            title: 图表标题
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        self._draw_skeleton(ax, joint_positions, parent_indices, joint_names, show_labels)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=self.elev, azim=self.azim)
        
        # 设置等比例坐标轴
        self._set_axes_equal(ax, joint_positions)
        
        plt.tight_layout()
        plt.show()
    
    def _draw_skeleton(
        self,
        ax: Axes3D,
        positions: np.ndarray,
        parent_indices: np.ndarray,
        names: Optional[List[str]] = None,
        show_labels: bool = False
    ) -> List:
        """绘制骨架到指定轴"""
        artists = []
        
        # 绘制关节点
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c='blue', s=50, alpha=0.8
        )
        artists.append(scatter)
        
        # 绘制骨骼连线
        for i, parent in enumerate(parent_indices):
            if parent >= 0:
                line, = ax.plot(
                    [positions[i, 0], positions[parent, 0]],
                    [positions[i, 1], positions[parent, 1]],
                    [positions[i, 2], positions[parent, 2]],
                    'b-', linewidth=2
                )
                artists.append(line)
        
        # 绘制标签
        if show_labels and names:
            for i, name in enumerate(names):
                ax.text(
                    positions[i, 0], positions[i, 1], positions[i, 2],
                    name, fontsize=8
                )
        
        return artists
    
    def _set_axes_equal(self, ax: Axes3D, positions: np.ndarray):
        """设置等比例坐标轴"""
        max_range = np.max([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]) / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def animate_motion(
        self,
        joint_positions_sequence: np.ndarray,
        parent_indices: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Motion Animation"
    ):
        """
        动画播放动作序列
        
        Args:
            joint_positions_sequence: 关节位置序列 (frames, joints, 3)
            parent_indices: 父节点索引
            save_path: 保存路径（可选）
            title: 动画标题
        """
        num_frames = len(joint_positions_sequence)
        
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title)
        self.ax.view_init(elev=self.elev, azim=self.azim)
        
        # 初始化
        positions = joint_positions_sequence[0]
        self._set_axes_equal(self.ax, joint_positions_sequence.reshape(-1, 3))
        
        # 创建初始绘图对象
        scatter = self.ax.scatter([], [], [], c='blue', s=50)
        lines = []
        for i, parent in enumerate(parent_indices):
            if parent >= 0:
                line, = self.ax.plot([], [], [], 'b-', linewidth=2)
                lines.append((i, parent, line))
        
        def init():
            scatter._offsets3d = ([], [], [])
            for _, _, line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [scatter] + [l[2] for l in lines]
        
        def update(frame_idx):
            positions = joint_positions_sequence[frame_idx]
            
            # 更新关节点
            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            
            # 更新骨骼连线
            for i, parent, line in lines:
                line.set_data(
                    [positions[i, 0], positions[parent, 0]],
                    [positions[i, 1], positions[parent, 1]]
                )
                line.set_3d_properties([positions[i, 2], positions[parent, 2]])
            
            self.ax.set_title(f"{title} - Frame {frame_idx + 1}/{num_frames}")
            
            return [scatter] + [l[2] for l in lines]
        
        self.animation = FuncAnimation(
            self.fig, update, frames=num_frames,
            init_func=init, blit=False,
            interval=1000 / self.fps
        )
        
        if save_path:
            save_path = Path(save_path)
            if save_path.suffix.lower() == '.gif':
                self.animation.save(str(save_path), writer='pillow', fps=self.fps)
            else:
                self.animation.save(str(save_path), writer='ffmpeg', fps=self.fps)
            print(f"Animation saved to: {save_path}")
        else:
            plt.show()
    
    def compare_motions(
        self,
        motion1: np.ndarray,
        motion2: np.ndarray,
        parent_indices: np.ndarray,
        labels: Tuple[str, str] = ("Original", "Generated"),
        frame_idx: int = 0
    ):
        """
        并排对比两个动作
        
        Args:
            motion1: 第一个动作序列 (frames, joints, 3)
            motion2: 第二个动作序列 (frames, joints, 3)
            parent_indices: 父节点索引
            labels: 标签
            frame_idx: 要显示的帧索引
        """
        fig = plt.figure(figsize=(self.figsize[0] * 2, self.figsize[1]))
        
        # 左侧：motion1
        ax1 = fig.add_subplot(121, projection='3d')
        self._draw_skeleton(ax1, motion1[frame_idx], parent_indices)
        ax1.set_title(f"{labels[0]} - Frame {frame_idx}")
        ax1.view_init(elev=self.elev, azim=self.azim)
        self._set_axes_equal(ax1, motion1[frame_idx])
        
        # 右侧：motion2
        ax2 = fig.add_subplot(122, projection='3d')
        self._draw_skeleton(ax2, motion2[frame_idx], parent_indices)
        ax2.set_title(f"{labels[1]} - Frame {frame_idx}")
        ax2.view_init(elev=self.elev, azim=self.azim)
        self._set_axes_equal(ax2, motion2[frame_idx])
        
        plt.tight_layout()
        plt.show()
    
    def plot_trajectory(
        self,
        root_positions: np.ndarray,
        title: str = "Root Trajectory"
    ):
        """
        绘制根节点轨迹
        
        Args:
            root_positions: 根节点位置序列 (frames, 3)
            title: 标题
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        ax.plot(
            root_positions[:, 0],
            root_positions[:, 1],
            root_positions[:, 2],
            'b-', linewidth=1, alpha=0.7
        )
        
        # 标记起点和终点
        ax.scatter(*root_positions[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(*root_positions[-1], c='red', s=100, marker='x', label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        ax.view_init(elev=self.elev, azim=self.azim)
        
        plt.tight_layout()
        plt.show()
