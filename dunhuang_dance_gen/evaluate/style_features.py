"""
敦煌舞风格特征提取与风格一致性评估模块

对应开题报告要求:
  - "从姿态序列中抽取具有敦煌舞特征的风格描述"
  - "构建简单的风格约束或风格判别模块"
  - "对敦煌风格的表达具有一定可控性"

风格特征维度:
  1. 上肢舒展度 — 肩-肘-手构成角度的统计分布
  2. 脊柱 S 曲线度 — 躯干链弯曲度分布 (敦煌舞 S 形身韵)
  3. 节奏停顿模式 — 零速度 / 低速度帧的分布与间隔
  4. 动作幅度分布 — 各关节运动范围的均值/方差
  5. 运动对称性 — 左右肢体旋转的相关度
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 敦煌舞 22 关节索引定义
# ============================================================
# 根据数据集 BVH 文件的实际骨骼层级:
# 0:Hips, 1:Chest, 2:Chest2, 3:Chest3, 4:Neck, 5:Head,
# 6:LeftCollar, 7:LeftShoulder, 8:LeftElbow, 9:LeftWrist,
# 10:RightCollar, 11:RightShoulder, 12:RightElbow, 13:RightWrist,
# 14:LeftUpLeg, 15:LeftLowLeg, 16:LeftFoot, 17:LeftToe,
# 18:RightUpLeg, 19:RightLowLeg, 20:RightFoot, 21:RightToe

JOINT_INDICES = {
    'hips': 0,
    'chest': 1, 'chest2': 2, 'chest3': 3,
    'neck': 4, 'head': 5,
    'left_collar': 6, 'left_shoulder': 7, 'left_elbow': 8, 'left_wrist': 9,
    'right_collar': 10, 'right_shoulder': 11, 'right_elbow': 12, 'right_wrist': 13,
    'left_upleg': 14, 'left_lowleg': 15, 'left_foot': 16, 'left_toe': 17,
    'right_upleg': 18, 'right_lowleg': 19, 'right_foot': 20, 'right_toe': 21,
}

# 功能分组
SPINE_CHAIN = [0, 1, 2, 3, 4]        # Hips → Chest → Chest2 → Chest3 → Neck
LEFT_ARM = [6, 7, 8, 9]               # LeftCollar → LeftShoulder → LeftElbow → LeftWrist
RIGHT_ARM = [10, 11, 12, 13]           # RightCollar → RightShoulder → RightElbow → RightWrist
LEFT_LEG = [14, 15, 16, 17]            # LeftUpLeg → LeftLowLeg → LeftFoot → LeftToe
RIGHT_LEG = [18, 19, 20, 21]           # RightUpLeg → RightLowLeg → RightFoot → RightToe
UPPER_BODY = SPINE_CHAIN + LEFT_ARM + RIGHT_ARM + [5]  # 含头
LOWER_BODY = LEFT_LEG + RIGHT_LEG


@dataclass
class DunhuangStyleProfile:
    """敦煌舞风格特征描述向量"""
    motion_name: str = ""
    
    # 1. 上肢舒展度
    left_arm_extension_mean: float = 0.0     # 左臂舒展角度均值
    left_arm_extension_std: float = 0.0      # 左臂舒展角度标准差
    right_arm_extension_mean: float = 0.0    # 右臂舒展角度均值
    right_arm_extension_std: float = 0.0
    arm_extension_range: float = 0.0         # 手臂活动范围 (max-min)
    
    # 2. 脊柱 S 曲线
    spine_curvature_mean: float = 0.0        # 脊柱弯曲度均值
    spine_curvature_std: float = 0.0         # 脊柱弯曲度标准差
    spine_curvature_max: float = 0.0         # 脊柱最大弯曲度
    spine_lateral_range: float = 0.0         # 脊柱侧弯幅度
    
    # 3. 节奏停顿模式
    pause_ratio: float = 0.0                 # 停顿帧占比 (%)
    mean_pause_duration: float = 0.0         # 平均停顿时长 (帧)
    pause_interval_mean: float = 0.0         # 停顿间隔均值 (帧)
    pause_interval_std: float = 0.0          # 停顿间隔标准差
    rhythm_regularity: float = 0.0           # 节奏规律性 (0~1, 越高越规律)
    
    # 4. 动作幅度分布
    upper_body_amplitude_mean: float = 0.0   # 上身动作幅度均值
    upper_body_amplitude_std: float = 0.0
    lower_body_amplitude_mean: float = 0.0   # 下身动作幅度均值
    lower_body_amplitude_std: float = 0.0
    upper_lower_ratio: float = 0.0           # 上下身幅度比 (敦煌舞特征: 上身>下身)
    
    # 5. 运动对称性
    arm_symmetry: float = 0.0                # 左右臂运动相关度 (0~1)
    leg_symmetry: float = 0.0                # 左右腿运动相关度
    overall_symmetry: float = 0.0            # 整体对称性
    
    def to_vector(self) -> np.ndarray:
        """将风格特征转为固定维度向量 (用于距离计算)"""
        return np.array([
            self.left_arm_extension_mean,
            self.left_arm_extension_std,
            self.right_arm_extension_mean,
            self.right_arm_extension_std,
            self.arm_extension_range,
            self.spine_curvature_mean,
            self.spine_curvature_std,
            self.spine_curvature_max,
            self.spine_lateral_range,
            self.pause_ratio,
            self.mean_pause_duration,
            self.pause_interval_mean,
            self.rhythm_regularity,
            self.upper_body_amplitude_mean,
            self.upper_body_amplitude_std,
            self.lower_body_amplitude_mean,
            self.lower_body_amplitude_std,
            self.upper_lower_ratio,
            self.arm_symmetry,
            self.leg_symmetry,
            self.overall_symmetry,
        ])
    
    def to_dict(self) -> Dict[str, float]:
        """转为字典，用于报告输出"""
        return {
            '左臂舒展度(均值°)': self.left_arm_extension_mean,
            '左臂舒展度(标准差°)': self.left_arm_extension_std,
            '右臂舒展度(均值°)': self.right_arm_extension_mean,
            '右臂舒展度(标准差°)': self.right_arm_extension_std,
            '手臂活动范围(°)': self.arm_extension_range,
            '脊柱弯曲度(均值°)': self.spine_curvature_mean,
            '脊柱弯曲度(标准差°)': self.spine_curvature_std,
            '脊柱最大弯曲度(°)': self.spine_curvature_max,
            '脊柱侧弯幅度(°)': self.spine_lateral_range,
            '停顿帧占比(%)': self.pause_ratio,
            '平均停顿时长(帧)': self.mean_pause_duration,
            '停顿间隔(均值帧)': self.pause_interval_mean,
            '节奏规律性(0~1)': self.rhythm_regularity,
            '上身幅度(均值°/s)': self.upper_body_amplitude_mean,
            '上身幅度(标准差)': self.upper_body_amplitude_std,
            '下身幅度(均值°/s)': self.lower_body_amplitude_mean,
            '下身幅度(标准差)': self.lower_body_amplitude_std,
            '上下身幅度比': self.upper_lower_ratio,
            '左右臂对称性': self.arm_symmetry,
            '左右腿对称性': self.leg_symmetry,
            '整体对称性': self.overall_symmetry,
        }


# ============================================================
# 风格特征提取器
# ============================================================

class DunhuangStyleExtractor:
    """
    敦煌舞风格特征提取器
    
    从旋转序列 (T, J, 3) 中提取 21 维风格特征向量，
    量化敦煌舞特有的上肢舒展、脊柱 S 曲线、节奏停顿、
    动作幅度和运动对称性。
    """
    
    def __init__(self, fps: float = 30.0, 
                 pause_velocity_threshold: float = 2.0,
                 min_pause_frames: int = 3):
        """
        Args:
            fps: 帧率
            pause_velocity_threshold: 停顿检测的速度阈值 (°/帧)
            min_pause_frames: 最小连续停顿帧数
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.pause_threshold = pause_velocity_threshold
        self.min_pause_frames = min_pause_frames
    
    def extract(self, rotations: np.ndarray, 
                motion_name: str = "unknown") -> DunhuangStyleProfile:
        """
        提取完整的风格特征描述
        
        Args:
            rotations: (T, J, 3) 或 (T, J*3) 旋转数据 (欧拉角, 度)
            motion_name: 动作名称
            
        Returns:
            DunhuangStyleProfile 风格特征描述
        """
        # 统一为 (T, J, 3) 格式
        if rotations.ndim == 2:
            T, D = rotations.shape
            J = D // 3
            rot = rotations.reshape(T, J, 3)
        else:
            rot = rotations
            T, J = rot.shape[:2]
        
        profile = DunhuangStyleProfile(motion_name=motion_name)
        
        # 1. 上肢舒展度
        self._compute_arm_extension(rot, profile)
        
        # 2. 脊柱 S 曲线
        self._compute_spine_curvature(rot, profile)
        
        # 3. 节奏停顿模式
        self._compute_rhythm_pattern(rot, profile)
        
        # 4. 动作幅度分布
        self._compute_amplitude_distribution(rot, profile)
        
        # 5. 运动对称性
        self._compute_symmetry(rot, profile)
        
        return profile
    
    def _compute_arm_extension(self, rot: np.ndarray, 
                                profile: DunhuangStyleProfile):
        """
        上肢舒展度: 量化肩-肘-手关节链的综合展开程度
        
        敦煌舞特征: 手臂经常做大幅度舒展动作 (如反弹琵琶、飞天)
        用肩+肘+腕三个关节的旋转幅度之和表示"舒展程度"
        """
        T = rot.shape[0]
        
        # 左臂: LeftShoulder(7) + LeftElbow(8) + LeftWrist(9)
        left_arm_joints = [7, 8, 9] if rot.shape[1] > 9 else LEFT_ARM[-3:]
        left_ext = np.zeros(T)
        for j in left_arm_joints:
            if j < rot.shape[1]:
                # 旋转幅度 = 三轴欧拉角的 L2 范数
                left_ext += np.sqrt(np.sum(rot[:, j, :] ** 2, axis=1))
        
        profile.left_arm_extension_mean = float(np.mean(left_ext))
        profile.left_arm_extension_std = float(np.std(left_ext))
        
        # 右臂: RightShoulder(11) + RightElbow(12) + RightWrist(13)
        right_arm_joints = [11, 12, 13] if rot.shape[1] > 13 else RIGHT_ARM[-3:]
        right_ext = np.zeros(T)
        for j in right_arm_joints:
            if j < rot.shape[1]:
                right_ext += np.sqrt(np.sum(rot[:, j, :] ** 2, axis=1))
        
        profile.right_arm_extension_mean = float(np.mean(right_ext))
        profile.right_arm_extension_std = float(np.std(right_ext))
        
        # 整体手臂活动范围
        all_ext = np.concatenate([left_ext, right_ext])
        profile.arm_extension_range = float(np.max(all_ext) - np.min(all_ext))
    
    def _compute_spine_curvature(self, rot: np.ndarray,
                                  profile: DunhuangStyleProfile):
        """
        脊柱 S 曲线度: 量化躯干链的弯曲程度
        
        敦煌舞特征: 身体常呈 S 形曲线 (三道弯)
        用脊柱链各关节旋转的相邻差异（弯曲度）衡量
        """
        T = rot.shape[0]
        
        # 脊柱链: Hips(0) → Chest(1) → Chest2(2) → Chest3(3) → Neck(4)
        spine_joints = SPINE_CHAIN
        valid_joints = [j for j in spine_joints if j < rot.shape[1]]
        
        if len(valid_joints) < 2:
            return
        
        # 计算相邻脊柱关节的旋转差 → 弯曲度
        curvatures = []
        for i in range(len(valid_joints) - 1):
            j1, j2 = valid_joints[i], valid_joints[i + 1]
            # 相邻关节的旋转差异
            diff = rot[:, j2, :] - rot[:, j1, :]
            curvature = np.sqrt(np.sum(diff ** 2, axis=1))
            curvatures.append(curvature)
        
        curvature_total = np.sum(curvatures, axis=0)  # (T,)
        
        profile.spine_curvature_mean = float(np.mean(curvature_total))
        profile.spine_curvature_std = float(np.std(curvature_total))
        profile.spine_curvature_max = float(np.max(curvature_total))
        
        # 侧弯幅度: 主要看 Z 轴 (左右侧弯) 的旋转范围
        lateral_rotations = []
        for j in valid_joints:
            lateral_rotations.append(rot[:, j, 2])  # Z 轴
        lateral = np.mean(lateral_rotations, axis=0)
        profile.spine_lateral_range = float(np.max(lateral) - np.min(lateral))
    
    def _compute_rhythm_pattern(self, rot: np.ndarray,
                                 profile: DunhuangStyleProfile):
        """
        节奏停顿模式: 检测零速度/低速度帧的分布
        
        敦煌舞特征: 动作中有明显的"提气-亮相"停顿节奏
        """
        T = rot.shape[0]
        
        if T < 3:
            return
        
        # 计算全身角速度 (每帧所有关节旋转变化量之和)
        rot_flat = rot.reshape(T, -1)
        vel = np.diff(rot_flat, axis=0)  # (T-1, J*3)
        frame_speed = np.sqrt(np.sum(vel ** 2, axis=1))  # (T-1,)
        
        # 停顿检测: 速度低于阈值 
        is_pause = frame_speed < self.pause_threshold
        
        # 计算停顿帧占比
        profile.pause_ratio = float(np.sum(is_pause) / len(is_pause) * 100)
        
        # 分析连续停顿段
        pause_segments = self._find_segments(is_pause)
        
        if len(pause_segments) > 0:
            durations = [end - start for start, end in pause_segments]
            profile.mean_pause_duration = float(np.mean(durations))
            
            # 停顿间隔
            if len(pause_segments) > 1:
                intervals = []
                for i in range(1, len(pause_segments)):
                    intervals.append(pause_segments[i][0] - pause_segments[i-1][1])
                profile.pause_interval_mean = float(np.mean(intervals))
                profile.pause_interval_std = float(np.std(intervals))
                
                # 节奏规律性 = 1 - CV (变异系数)
                if profile.pause_interval_mean > 0:
                    cv = profile.pause_interval_std / profile.pause_interval_mean
                    profile.rhythm_regularity = float(max(0, 1 - cv))
    
    def _find_segments(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """找出连续 True 的段 [(start, end), ...]"""
        segments = []
        in_segment = False
        start = 0
        
        for i, v in enumerate(mask):
            if v and not in_segment:
                start = i
                in_segment = True
            elif not v and in_segment:
                if i - start >= self.min_pause_frames:
                    segments.append((start, i))
                in_segment = False
        
        if in_segment and len(mask) - start >= self.min_pause_frames:
            segments.append((start, len(mask)))
        
        return segments
    
    def _compute_amplitude_distribution(self, rot: np.ndarray,
                                         profile: DunhuangStyleProfile):
        """
        动作幅度分布: 上身 vs 下身的运动幅度对比
        
        敦煌舞特征: 上身动作丰富、下盘稳健 → 上下身幅度比 > 1
        """
        T = rot.shape[0]
        
        if T < 2:
            return
        
        # 角速度 (每帧旋转变化量)
        vel = np.diff(rot, axis=0)  # (T-1, J, 3)
        joint_speed = np.sqrt(np.sum(vel ** 2, axis=2))  # (T-1, J)
        
        # 上身关节的动作幅度
        upper_joints = [j for j in UPPER_BODY if j < rot.shape[1]]
        if upper_joints:
            upper_speed = joint_speed[:, upper_joints]
            profile.upper_body_amplitude_mean = float(np.mean(upper_speed))
            profile.upper_body_amplitude_std = float(np.std(upper_speed))
        
        # 下身关节的动作幅度
        lower_joints = [j for j in LOWER_BODY if j < rot.shape[1]]
        if lower_joints:
            lower_speed = joint_speed[:, lower_joints]
            profile.lower_body_amplitude_mean = float(np.mean(lower_speed))
            profile.lower_body_amplitude_std = float(np.std(lower_speed))
        
        # 上/下身幅度比
        if profile.lower_body_amplitude_mean > 0.01:
            profile.upper_lower_ratio = (
                profile.upper_body_amplitude_mean / profile.lower_body_amplitude_mean
            )
        else:
            profile.upper_lower_ratio = float('inf')
    
    def _compute_symmetry(self, rot: np.ndarray,
                           profile: DunhuangStyleProfile):
        """
        运动对称性: 左右肢体旋转序列的Pearson相关系数
        
        敦煌舞特征: 部分动作高度对称 (如菩萨立像), 
        部分刻意不对称 (如反弹琵琶)
        """
        T = rot.shape[0]
        
        if T < 5:
            return
        
        def _correlation(joints_a, joints_b):
            """计算两组关节旋转序列的平均相关系数"""
            corrs = []
            for ja, jb in zip(joints_a, joints_b):
                if ja < rot.shape[1] and jb < rot.shape[1]:
                    a = rot[:, ja, :].flatten()
                    b = rot[:, jb, :].flatten()
                    if np.std(a) > 1e-6 and np.std(b) > 1e-6:
                        corr = np.corrcoef(a, b)[0, 1]
                        if not np.isnan(corr):
                            corrs.append(abs(corr))
            return float(np.mean(corrs)) if corrs else 0.0
        
        # 左右臂对称性
        profile.arm_symmetry = _correlation(
            [7, 8, 9],    # LeftShoulder, LeftElbow, LeftWrist
            [11, 12, 13]  # RightShoulder, RightElbow, RightWrist
        )
        
        # 左右腿对称性
        profile.leg_symmetry = _correlation(
            [14, 15, 16],  # LeftUpLeg, LeftLowLeg, LeftFoot
            [18, 19, 20]   # RightUpLeg, RightLowLeg, RightFoot
        )
        
        # 整体对称性 (加权平均, 臂权重更高因为敦煌舞上身动作更重要)
        profile.overall_symmetry = 0.6 * profile.arm_symmetry + 0.4 * profile.leg_symmetry


# ============================================================
# 风格一致性评估 (跨舞段对比)
# ============================================================

class StyleConsistencyEvaluator:
    """
    跨舞段风格一致性评估器
    
    用同一组风格指标评估不同舞段的生成结果,
    证明它们在风格维度上保持一致的"敦煌特征"。
    """
    
    def __init__(self, extractor: Optional[DunhuangStyleExtractor] = None):
        self.extractor = extractor or DunhuangStyleExtractor()
        self.profiles: Dict[str, DunhuangStyleProfile] = {}
    
    def add_motion(self, name: str, rotations: np.ndarray):
        """添加一段动作的风格特征"""
        self.profiles[name] = self.extractor.extract(rotations, name)
    
    def compute_style_distance(self, name_a: str, name_b: str,
                                global_mean: np.ndarray = None,
                                global_std: np.ndarray = None) -> float:
        """
        计算两段动作之间的风格距离 (归一化欧氏距离)
        
        距离越小 → 风格越一致
        """
        if name_a not in self.profiles or name_b not in self.profiles:
            return float('nan')
        
        vec_a = self.profiles[name_a].to_vector()
        vec_b = self.profiles[name_b].to_vector()
        
        # 使用全局统计量归一化
        if global_std is not None:
            std = global_std.copy()
            std[std < 1e-8] = 1.0
            if global_mean is not None:
                vec_a = (vec_a - global_mean) / std
                vec_b = (vec_b - global_mean) / std
            else:
                vec_a = vec_a / std
                vec_b = vec_b / std
        
        return float(np.sqrt(np.sum((vec_a - vec_b) ** 2)))
    
    def compute_consistency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        计算所有已添加动作之间的风格距离矩阵
        
        使用全局均值/标准差归一化，确保距离具有可比性
        
        Returns:
            distance_matrix: (N, N) 风格距离矩阵
            names: 名称列表
        """
        names = sorted(self.profiles.keys())
        n = len(names)
        
        # 计算全局统计量 (跨所有 profile)
        all_vecs = np.stack([self.profiles[name].to_vector() for name in names])
        global_mean = np.mean(all_vecs, axis=0)
        global_std = np.std(all_vecs, axis=0)
        
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = self.compute_style_distance(
                        names[i], names[j], global_mean, global_std)
        
        return matrix, names
    
    def compute_style_preservation_score(self, 
                                          original_name: str, 
                                          generated_name: str) -> Dict[str, float]:
        """
        计算风格保持得分: 生成动作相对于原始动作的风格保持程度
        
        Returns:
            各维度的保持率 (0~1, 越高越好) 和综合得分
        """
        if original_name not in self.profiles or generated_name not in self.profiles:
            return {'风格保持综合得分': 0.0}
        
        orig = self.profiles[original_name]
        gen = self.profiles[generated_name]
        
        scores = {}
        
        # 各维度保持率 (使用相对误差的倒数)
        def _preservation(orig_val, gen_val, name):
            if abs(orig_val) < 1e-6:
                return 1.0 if abs(gen_val) < 1e-6 else 0.0
            rel_error = abs(gen_val - orig_val) / (abs(orig_val) + 1e-8)
            score = max(0, 1 - rel_error)
            scores[name] = round(score, 4)
            return score
        
        dims = []
        dims.append(_preservation(orig.left_arm_extension_mean, gen.left_arm_extension_mean, '上肢舒展度保持'))
        dims.append(_preservation(orig.spine_curvature_mean, gen.spine_curvature_mean, '脊柱曲线保持'))
        dims.append(_preservation(orig.pause_ratio, gen.pause_ratio, '节奏停顿保持'))
        dims.append(_preservation(orig.upper_lower_ratio, gen.upper_lower_ratio, '上下身比例保持'))
        dims.append(_preservation(orig.overall_symmetry, gen.overall_symmetry, '对称性保持'))
        
        # 综合得分 (加权平均)
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]
        overall = sum(w * d for w, d in zip(weights, dims))
        scores['风格保持综合得分'] = round(overall, 4)
        
        return scores
    
    def generate_report(self) -> str:
        """生成跨舞段风格一致性分析报告"""
        if len(self.profiles) < 2:
            return "需要至少 2 段动作才能进行风格一致性分析"
        
        lines = [
            "## 跨舞段敦煌舞风格一致性分析",
            "",
            "### 风格特征汇总",
            "",
            "| 动作 | 上肢舒展度 | 脊柱弯曲度 | 停顿占比(%) | 上下身比 | 整体对称性 |",
            "|------|-----------|-----------|------------|---------|-----------|",
        ]
        
        for name in sorted(self.profiles.keys()):
            p = self.profiles[name]
            lines.append(
                f"| {name} | {p.left_arm_extension_mean:.1f}±{p.left_arm_extension_std:.1f} "
                f"| {p.spine_curvature_mean:.1f}±{p.spine_curvature_std:.1f} "
                f"| {p.pause_ratio:.1f} "
                f"| {p.upper_lower_ratio:.2f} "
                f"| {p.overall_symmetry:.3f} |"
            )
        
        # 风格距离矩阵
        matrix, names = self.compute_consistency_matrix()
        
        lines.extend([
            "",
            "### 风格距离矩阵",
            "",
            "| " + " | ".join([""] + [n[:12] for n in names]) + " |",
            "| " + " | ".join(["---"] * (len(names) + 1)) + " |",
        ])
        
        for i, name in enumerate(names):
            row = [name[:12]]
            for j in range(len(names)):
                if i == j:
                    row.append("-")
                else:
                    row.append(f"{matrix[i,j]:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        
        # 一致性结论
        if len(names) >= 2:
            off_diag = matrix[np.triu_indices_from(matrix, k=1)]
            avg_dist = float(np.mean(off_diag))
            lines.extend([
                "",
                f"**跨舞段平均风格距离**: {avg_dist:.2f}",
                "",
                "- 距离越小表示风格越一致",
                "- 同一舞种（如琵琶伎乐1 vs 琵琶伎乐2）的距离应显著小于跨舞种距离",
            ])
        
        return "\n".join(lines)
