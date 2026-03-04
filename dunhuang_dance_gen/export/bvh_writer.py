"""
BVH Writer for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - BVH写入器

将动作数据导出为标准 BVH 格式
"""

import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path


class BVHWriter:
    """BVH 文件写入器"""
    
    def __init__(
        self,
        rotation_order: str = 'xyz',
        frame_time: float = 1.0 / 30.0,
        precision: int = 6
    ):
        """
        初始化 BVH 写入器
        
        Args:
            rotation_order: 旋转顺序 ('xyz', 'zxy', 'zyx' 等)
            frame_time: 帧时间间隔
            precision: 浮点数精度
        """
        self.rotation_order = rotation_order.upper()
        self.frame_time = frame_time
        self.precision = precision
    
    def write(
        self,
        filepath: str,
        joint_names: List[str],
        parent_indices: np.ndarray,
        offsets: np.ndarray,
        positions: np.ndarray,
        rotations: np.ndarray,
        frame_time: Optional[float] = None
    ) -> str:
        """
        写入 BVH 文件
        
        Args:
            filepath: 输出文件路径
            joint_names: 关节名称列表
            parent_indices: 父节点索引数组
            offsets: 各关节偏移 (joints, 3)
            positions: 根节点位置 (frames, 3)
            rotations: 各关节旋转 (frames, joints, 3) 欧拉角
            frame_time: 帧时间间隔
            
        Returns:
            str: 写入的文件路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if frame_time is None:
            frame_time = self.frame_time
        
        num_frames = len(positions)
        num_joints = len(joint_names)
        
        # 构建骨架层级结构
        children = self._build_children_map(parent_indices)
        
        lines = []
        
        # 写入 HIERARCHY 部分
        lines.append("HIERARCHY")
        joint_order = []  # 记录关节写入顺序
        self._write_joint(lines, 0, joint_names, offsets, children, joint_order, prefix="")
        
        # 写入 MOTION 部分
        lines.append("MOTION")
        lines.append(f"Frames: {num_frames}")
        lines.append(f"Frame Time: {frame_time:.8f}")
        
        # 写入帧数据
        for frame_idx in range(num_frames):
            frame_data = []
            
            # 按照骨架写入顺序遍历关节
            for j, joint_idx in enumerate(joint_order):
                if joint_idx == 0:
                    # 根节点：位置 + 旋转
                    frame_data.extend([
                        f"{positions[frame_idx, 0]:.{self.precision}f}",
                        f"{positions[frame_idx, 1]:.{self.precision}f}",
                        f"{positions[frame_idx, 2]:.{self.precision}f}"
                    ])
                
                # 所有关节的旋转
                rot = rotations[frame_idx, joint_idx]
                for axis in self.rotation_order.lower():
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    frame_data.append(f"{rot[axis_idx]:.{self.precision}f}")
            
            lines.append(" ".join(frame_data))
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        return str(filepath)
    
    def _build_children_map(self, parent_indices: np.ndarray) -> dict:
        """构建子节点映射"""
        children = {i: [] for i in range(len(parent_indices))}
        
        for i, parent in enumerate(parent_indices):
            if parent >= 0:
                children[parent].append(i)
        
        return children
    
    def _write_joint(
        self, 
        lines: List[str], 
        joint_idx: int,
        joint_names: List[str],
        offsets: np.ndarray,
        children: dict,
        joint_order: List[int],
        prefix: str
    ):
        """递归写入关节定义"""
        name = joint_names[joint_idx]
        offset = offsets[joint_idx]
        child_list = children[joint_idx]
        
        # 记录写入顺序
        joint_order.append(joint_idx)
        
        # 关节类型
        if joint_idx == 0:
            lines.append(f"{prefix}ROOT {name}")
        else:
            lines.append(f"{prefix}JOINT {name}")
        
        lines.append(f"{prefix}{{")
        
        # 偏移
        lines.append(f"{prefix}\tOFFSET {offset[0]:.{self.precision}f} {offset[1]:.{self.precision}f} {offset[2]:.{self.precision}f}")
        
        # 通道定义
        if joint_idx == 0:
            lines.append(f"{prefix}\tCHANNELS 6 Xposition Yposition Zposition {self.rotation_order[0]}rotation {self.rotation_order[1]}rotation {self.rotation_order[2]}rotation")
        else:
            lines.append(f"{prefix}\tCHANNELS 3 {self.rotation_order[0]}rotation {self.rotation_order[1]}rotation {self.rotation_order[2]}rotation")
        
        # 子关节或末端
        if child_list:
            for child_idx in child_list:
                self._write_joint(lines, child_idx, joint_names, offsets, children, joint_order, prefix + "\t")
        else:
            # 末端节点
            lines.append(f"{prefix}\tEnd Site")
            lines.append(f"{prefix}\t{{")
            lines.append(f"{prefix}\t\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{prefix}\t}}")
        
        lines.append(f"{prefix}}}")
    
    def write_from_bvhdata(self, filepath: str, data) -> str:
        """
        从 BVHData 对象写入文件
        
        Args:
            filepath: 输出路径
            data: BVHData 对象
            
        Returns:
            写入的文件路径
        """
        return self.write(
            filepath=filepath,
            joint_names=data.joint_names,
            parent_indices=data.parent_indices,
            offsets=data.offsets,
            positions=data.positions,
            rotations=data.rotations,
            frame_time=data.frame_time
        )


def save_bvh(
    filepath: str,
    joint_names: List[str],
    parent_indices: np.ndarray,
    offsets: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    frame_time: float = 1.0 / 30.0
) -> str:
    """
    便捷函数：保存 BVH 文件
    
    Returns:
        写入的文件路径
    """
    writer = BVHWriter(frame_time=frame_time)
    return writer.write(
        filepath, joint_names, parent_indices, 
        offsets, positions, rotations, frame_time
    )
