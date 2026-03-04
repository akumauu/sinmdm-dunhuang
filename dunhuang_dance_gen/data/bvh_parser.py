"""
BVH Parser for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - BVH解析器

基于 SinMDM 的 bvh_io.py 实现的轻量级 BVH 解析器
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path


@dataclass
class BVHData:
    """BVH 数据结构"""
    # 骨架信息
    joint_names: List[str] = field(default_factory=list)
    parent_indices: np.ndarray = None  # 父节点索引
    offsets: np.ndarray = None  # 各关节相对父节点的偏移
    
    # 动作数据
    positions: np.ndarray = None  # 根节点位置 (frames, 3)
    rotations: np.ndarray = None  # 各关节旋转 (frames, joints, 3) 欧拉角
    
    # 时序信息
    frame_time: float = 1.0 / 30.0  # 帧时间间隔
    num_frames: int = 0
    num_joints: int = 0
    
    # 旋转顺序
    rotation_order: str = 'xyz'
    
    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time if self.frame_time > 0 else 30.0
    
    @property
    def duration(self) -> float:
        """动画时长（秒）"""
        return self.num_frames * self.frame_time
    
    def get_joint_index(self, name: str) -> int:
        """根据名称获取关节索引"""
        try:
            return self.joint_names.index(name)
        except ValueError:
            return -1
    
    def get_children(self, joint_idx: int) -> List[int]:
        """获取指定关节的子关节索引"""
        return [i for i, p in enumerate(self.parent_indices) if p == joint_idx]


class BVHParser:
    """BVH 文件解析器"""
    
    CHANNEL_MAP = {
        'Xrotation': 'x', 'Yrotation': 'y', 'Zrotation': 'z',
        'Xposition': 'px', 'Yposition': 'py', 'Zposition': 'pz'
    }
    
    def __init__(self):
        self.data: Optional[BVHData] = None
    
    def parse(self, filepath: str) -> BVHData:
        """
        解析 BVH 文件
        
        Args:
            filepath: BVH 文件路径
            
        Returns:
            BVHData: 解析后的数据
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"BVH file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分离骨架定义和动作数据
        if 'MOTION' not in content:
            raise ValueError("Invalid BVH file: missing MOTION section")
        
        hierarchy_part, motion_part = content.split('MOTION')
        
        # 解析骨架
        joint_names, parent_indices, offsets, channel_info = self._parse_hierarchy(hierarchy_part)
        
        # 解析动作
        positions, rotations, frame_time, num_frames = self._parse_motion(
            motion_part, len(joint_names), channel_info
        )
        
        # 构建数据对象
        self.data = BVHData(
            joint_names=joint_names,
            parent_indices=np.array(parent_indices, dtype=np.int32),
            offsets=np.array(offsets, dtype=np.float32),
            positions=positions,
            rotations=rotations,
            frame_time=frame_time,
            num_frames=num_frames,
            num_joints=len(joint_names)
        )
        
        return self.data
    
    def _parse_hierarchy(self, content: str) -> Tuple[List[str], List[int], List[List[float]], List[Dict]]:
        """解析骨架层级结构"""
        joint_names = []
        parent_indices = []
        offsets = []
        channel_info = []  # 每个关节的通道信息
        
        active_joint = -1
        end_site = False
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line == 'HIERARCHY':
                continue
            
            # ROOT 节点
            root_match = re.match(r'ROOT\s+(\S+)', line)
            if root_match:
                joint_names.append(root_match.group(1))
                parent_indices.append(-1)
                offsets.append([0.0, 0.0, 0.0])
                active_joint = len(joint_names) - 1
                continue
            
            # JOINT 节点
            joint_match = re.match(r'JOINT\s+(\S+)', line)
            if joint_match:
                joint_names.append(joint_match.group(1))
                parent_indices.append(active_joint)
                offsets.append([0.0, 0.0, 0.0])
                active_joint = len(joint_names) - 1
                continue
            
            # End Site
            if 'End Site' in line:
                end_site = True
                continue
            
            # 大括号
            if line == '{':
                continue
            if line == '}':
                if end_site:
                    end_site = False
                else:
                    active_joint = parent_indices[active_joint] if active_joint >= 0 else -1
                continue
            
            # OFFSET
            offset_match = re.match(r'OFFSET\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)', line)
            if offset_match and not end_site:
                offsets[active_joint] = [float(x) for x in offset_match.groups()]
                continue
            
            # CHANNELS
            channel_match = re.match(r'CHANNELS\s+(\d+)\s+(.+)', line)
            if channel_match:
                num_channels = int(channel_match.group(1))
                channels = channel_match.group(2).split()
                channel_info.append({
                    'joint_idx': active_joint,
                    'num_channels': num_channels,
                    'channels': channels
                })
                continue
        
        return joint_names, parent_indices, offsets, channel_info
    
    def _parse_motion(self, content: str, num_joints: int, channel_info: List[Dict]) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """解析动作数据"""
        lines = content.strip().split('\n')
        
        num_frames = 0
        frame_time = 1.0 / 30.0
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            frames_match = re.match(r'Frames:\s*(\d+)', line)
            if frames_match:
                num_frames = int(frames_match.group(1))
                continue
            
            frametime_match = re.match(r'Frame Time:\s*([\d\.]+)', line)
            if frametime_match:
                frame_time = float(frametime_match.group(1))
                data_start_idx = i + 1
                break
        
        # 解析帧数据
        positions = np.zeros((num_frames, 3), dtype=np.float32)
        rotations = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
        
        frame_idx = 0
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            values = [float(x) for x in line.split()]
            
            # 根据通道信息分配数据
            value_idx = 0
            for ch_info in channel_info:
                joint_idx = ch_info['joint_idx']
                for channel in ch_info['channels']:
                    if value_idx >= len(values):
                        break
                    
                    if channel == 'Xposition':
                        positions[frame_idx, 0] = values[value_idx]
                    elif channel == 'Yposition':
                        positions[frame_idx, 1] = values[value_idx]
                    elif channel == 'Zposition':
                        positions[frame_idx, 2] = values[value_idx]
                    elif channel == 'Xrotation':
                        rotations[frame_idx, joint_idx, 0] = values[value_idx]
                    elif channel == 'Yrotation':
                        rotations[frame_idx, joint_idx, 1] = values[value_idx]
                    elif channel == 'Zrotation':
                        rotations[frame_idx, joint_idx, 2] = values[value_idx]
                    
                    value_idx += 1
            
            frame_idx += 1
            if frame_idx >= num_frames:
                break
        
        return positions, rotations, frame_time, num_frames
    
    def load(self, filepath: str) -> BVHData:
        """加载 BVH 文件（parse 的别名）"""
        return self.parse(filepath)


def load_bvh(filepath: str) -> BVHData:
    """
    便捷函数：加载 BVH 文件
    
    Args:
        filepath: BVH 文件路径
        
    Returns:
        BVHData: 解析后的数据
    """
    parser = BVHParser()
    return parser.parse(filepath)
