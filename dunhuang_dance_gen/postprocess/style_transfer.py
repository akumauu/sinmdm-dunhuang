"""
敦煌舞风格迁移与可控风格约束模块

功能:
  1. 风格迁移 — 从参考动作提取风格, 迁移到目标动作
  2. 风格约束 — 根据目标风格参数修改动作关节旋转
  3. 风格混合 — 两段动作按权重混合, 产生新序列

输出: 修改后的旋转序列 (T, J, 3), 可直接导出 BVH
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from dunhuang_dance_gen.evaluate.style_features import (
    DunhuangStyleExtractor, DunhuangStyleProfile,
    SPINE_CHAIN, LEFT_ARM, RIGHT_ARM, UPPER_BODY, LOWER_BODY,
    LEFT_LEG, RIGHT_LEG
)


def _apply_symmetry_scale(rot: np.ndarray, pairs: List[Tuple[int, int]], diff_scale: float) -> np.ndarray:
    """Scale left-right differences while preserving the average pose."""
    for left_idx, right_idx in pairs:
        if left_idx >= rot.shape[1] or right_idx >= rot.shape[1]:
            continue
        left = rot[:, left_idx, :].copy()
        right = rot[:, right_idx, :].copy()
        center = 0.5 * (left + right)
        diff = 0.5 * (left - right)
        rot[:, left_idx, :] = center + diff * diff_scale
        rot[:, right_idx, :] = center - diff * diff_scale
    return rot


@dataclass
class StyleTransferConfig:
    """风格迁移参数配置"""
    # 各维度迁移强度 (0.0 = 不迁移, 1.0 = 完全迁移)
    arm_extension_strength: float = 0.5       # 上肢舒展度
    spine_curvature_strength: float = 0.5     # 脊柱 S 曲线
    rhythm_strength: float = 0.3              # 节奏停顿
    amplitude_strength: float = 0.4           # 动作幅度
    symmetry_strength: float = 0.3            # 对称性
    
    # 全局混合系数
    global_strength: float = 1.0              # 总体强度 (乘到所有维度上)
    
    # 平滑参数
    smooth_transition: bool = True            # 是否平滑过渡
    transition_frames: int = 10               # 过渡帧数


@dataclass
class StyleTransferResult:
    """风格迁移结果"""
    rotations: np.ndarray                     # 迁移后的旋转 (T, J, 3)
    positions: np.ndarray                     # 位置 (可能微调)
    source_style: DunhuangStyleProfile        # 源动作风格
    target_style: DunhuangStyleProfile        # 目标风格 (迁移目标)
    result_style: DunhuangStyleProfile        # 迁移后风格
    config: StyleTransferConfig
    
    def summary(self) -> str:
        lines = ["=== 风格迁移结果 ==="]
        lines.append(f"源风格  - 上肢舒展: {self.source_style.left_arm_extension_mean:.1f}°")
        lines.append(f"目标风格 - 上肢舒展: {self.target_style.left_arm_extension_mean:.1f}°")
        lines.append(f"结果    - 上肢舒展: {self.result_style.left_arm_extension_mean:.1f}°")
        lines.append(f"源风格  - 脊柱弯曲: {self.source_style.spine_curvature_mean:.1f}°")
        lines.append(f"目标风格 - 脊柱弯曲: {self.target_style.spine_curvature_mean:.1f}°")
        lines.append(f"结果    - 脊柱弯曲: {self.result_style.spine_curvature_mean:.1f}°")
        return "\n".join(lines)


class DunhuangStyleTransfer:
    """
    敦煌舞风格迁移器
    
    从参考动作提取风格特征, 通过直接调整旋转数据
    将风格迁移到目标动作序列, 输出可导出的 BVH 数据。
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.extractor = DunhuangStyleExtractor(fps=fps)
    
    def transfer(self,
                 source_rotations: np.ndarray,
                 source_positions: np.ndarray,
                 reference_rotations: np.ndarray,
                 config: Optional[StyleTransferConfig] = None) -> StyleTransferResult:
        """
        风格迁移: 将 reference 的风格迁移到 source 上
        
        Args:
            source_rotations: (T, J, 3) 源动作旋转 (要被修改的)
            source_positions: (T, 3) 源动作位置
            reference_rotations: (T2, J, 3) 参考动作旋转 (风格来源)
            config: 迁移配置
            
        Returns:
            StyleTransferResult 含迁移后的旋转/位置数据
        """
        cfg = config or StyleTransferConfig()
        
        # 提取风格特征
        src_profile = self.extractor.extract(source_rotations, "source")
        ref_profile = self.extractor.extract(reference_rotations, "reference")
        
        # 复制数据
        rot = source_rotations.copy()
        pos = source_positions.copy()
        T, J = rot.shape[:2]
        
        g = cfg.global_strength
        
        # 1. 上肢舒展度迁移
        if cfg.arm_extension_strength > 0:
            s = cfg.arm_extension_strength * g
            rot = self._transfer_arm_extension(rot, src_profile, ref_profile, s)
        
        # 2. 脊柱 S 曲线迁移
        if cfg.spine_curvature_strength > 0:
            s = cfg.spine_curvature_strength * g
            rot = self._transfer_spine_curvature(rot, src_profile, ref_profile, s)
        
        # 3. 节奏停顿迁移
        if cfg.rhythm_strength > 0:
            s = cfg.rhythm_strength * g
            rot = self._transfer_rhythm(rot, source_rotations, reference_rotations, s)
        
        # 4. 动作幅度迁移
        if cfg.amplitude_strength > 0:
            s = cfg.amplitude_strength * g
            rot = self._transfer_amplitude(rot, src_profile, ref_profile, s)

        # 5. 左右对称性迁移
        if cfg.symmetry_strength > 0:
            s = cfg.symmetry_strength * g
            rot = self._transfer_symmetry(rot, src_profile, ref_profile, s)
        
        # 6. 平滑过渡
        if cfg.smooth_transition:
            rot = self._smooth_result(rot, source_rotations, cfg.transition_frames)
        
        # 提取结果风格
        result_profile = self.extractor.extract(rot, "result")
        
        return StyleTransferResult(
            rotations=rot,
            positions=pos,
            source_style=src_profile,
            target_style=ref_profile,
            result_style=result_profile,
            config=cfg,
        )
    
    def _transfer_arm_extension(self, rot, src_profile, ref_profile, strength):
        """
        迁移上肢舒展度:
        计算参考与源的臂展比例, 按比例缩放手臂关节旋转
        """
        T, J = rot.shape[:2]
        
        # 左臂缩放
        if src_profile.left_arm_extension_mean > 1e-3:
            ratio = ref_profile.left_arm_extension_mean / src_profile.left_arm_extension_mean
            scale = 1.0 + (ratio - 1.0) * strength
            for j in [7, 8, 9]:  # LeftShoulder, LeftElbow, LeftWrist
                if j < J:
                    rot[:, j, :] *= scale
        
        # 右臂缩放
        if src_profile.right_arm_extension_mean > 1e-3:
            ratio = ref_profile.right_arm_extension_mean / src_profile.right_arm_extension_mean
            scale = 1.0 + (ratio - 1.0) * strength
            for j in [11, 12, 13]:  # RightShoulder, RightElbow, RightWrist
                if j < J:
                    rot[:, j, :] *= scale
        
        return rot
    
    def _transfer_spine_curvature(self, rot, src_profile, ref_profile, strength):
        """
        迁移脊柱 S 曲线度:
        调整脊柱链关节间差异, 增强或减弱弯曲
        """
        T, J = rot.shape[:2]
        
        if src_profile.spine_curvature_mean > 1e-3:
            ratio = ref_profile.spine_curvature_mean / src_profile.spine_curvature_mean
            scale = 1.0 + (ratio - 1.0) * strength
            
            # 对脊柱链的相邻关节旋转差异进行缩放
            spine_joints = [j for j in SPINE_CHAIN if j < J]
            if len(spine_joints) >= 2:
                # 计算脊柱各关节相对于 Hips 的偏移
                base_rot = rot[:, spine_joints[0], :].copy()
                for k in range(1, len(spine_joints)):
                    j = spine_joints[k]
                    diff = rot[:, j, :] - base_rot
                    rot[:, j, :] = base_rot + diff * scale
        
        # 侧弯调整
        if src_profile.spine_lateral_range > 1e-3 and ref_profile.spine_lateral_range > 1e-3:
            lat_ratio = ref_profile.spine_lateral_range / src_profile.spine_lateral_range
            lat_scale = 1.0 + (lat_ratio - 1.0) * strength * 0.5
            for j in [j for j in SPINE_CHAIN if j < J]:
                rot[:, j, 2] *= lat_scale  # Z 轴 = 侧弯
        
        return rot
    
    def _transfer_rhythm(self, rot, src_rot, ref_rot, strength):
        """
        迁移节奏停顿模式:
        检测参考动作的停顿模式, 在源动作对应位置插入/增强停顿
        """
        T, J = rot.shape[:2]
        T_ref = ref_rot.shape[0]
        
        # 计算参考动作的帧速度
        ref_flat = ref_rot.reshape(T_ref, -1)
        ref_vel = np.sqrt(np.sum(np.diff(ref_flat, axis=0) ** 2, axis=1))
        
        # 检测参考的停顿帧 (低速度)
        ref_threshold = np.percentile(ref_vel, 20)  # 最低 20% 速度的帧
        ref_pause_mask = ref_vel < ref_threshold
        
        # 计算参考的停顿比例
        ref_pause_ratio = np.sum(ref_pause_mask) / len(ref_pause_mask)
        
        # 计算源动作的帧速度
        src_flat = rot.reshape(T, -1)
        src_vel = np.sqrt(np.sum(np.diff(src_flat, axis=0) ** 2, axis=1))
        
        # 找出源动作中速度最低的帧 (按参考的停顿比例)
        target_pause_count = int(ref_pause_ratio * (T - 1) * strength)
        if target_pause_count > 0:
            # 选择速度最低的帧进行减速
            lowest_idx = np.argsort(src_vel)[:target_pause_count]
            
            # 在这些帧附近降低运动速度 (平滑插值)
            for idx in lowest_idx:
                if idx > 0 and idx < T - 1:
                    # 向相邻帧插值, 降低变化量
                    alpha = 0.3 * strength  # 减速程度
                    rot[idx, :, :] = (1 - alpha) * rot[idx, :, :] + alpha * rot[idx - 1, :, :]
        
        return rot
    
    def _transfer_amplitude(self, rot, src_profile, ref_profile, strength):
        """
        迁移动作幅度分布:
        调整上身/下身的运动幅度比例
        """
        T, J = rot.shape[:2]
        
        # 上身幅度调整
        if src_profile.upper_body_amplitude_mean > 1e-3:
            ratio = ref_profile.upper_body_amplitude_mean / src_profile.upper_body_amplitude_mean
            scale = 1.0 + (ratio - 1.0) * strength
            upper_joints = [j for j in UPPER_BODY if j < J]
            for j in upper_joints:
                # 只缩放变化量, 保持基准姿态
                mean_rot = np.mean(rot[:, j, :], axis=0, keepdims=True)
                rot[:, j, :] = mean_rot + (rot[:, j, :] - mean_rot) * scale
        
        # 下身幅度调整
        if src_profile.lower_body_amplitude_mean > 1e-3:
            ratio = ref_profile.lower_body_amplitude_mean / src_profile.lower_body_amplitude_mean
            scale = 1.0 + (ratio - 1.0) * strength
            lower_joints = [j for j in LOWER_BODY if j < J]
            for j in lower_joints:
                mean_rot = np.mean(rot[:, j, :], axis=0, keepdims=True)
                rot[:, j, :] = mean_rot + (rot[:, j, :] - mean_rot) * scale
        
        return rot

    def _transfer_symmetry(self, rot, src_profile, ref_profile, strength):
        """
        迁移左右对称性:
        根据参考动作的对称性，缩放左右肢体的差异项。
        """
        arm_src_asym = max(1e-3, 1.0 - src_profile.arm_symmetry)
        arm_ref_asym = max(0.0, 1.0 - ref_profile.arm_symmetry)
        arm_ratio = arm_ref_asym / arm_src_asym
        arm_scale = 1.0 + (arm_ratio - 1.0) * strength
        rot = _apply_symmetry_scale(rot, [(7, 11), (8, 12), (9, 13)], arm_scale)

        leg_src_asym = max(1e-3, 1.0 - src_profile.leg_symmetry)
        leg_ref_asym = max(0.0, 1.0 - ref_profile.leg_symmetry)
        leg_ratio = leg_ref_asym / leg_src_asym
        leg_scale = 1.0 + (leg_ratio - 1.0) * strength
        rot = _apply_symmetry_scale(rot, [(14, 18), (15, 19), (16, 20)], leg_scale)
        return rot
    
    def _smooth_result(self, rot, original_rot, transition_frames):
        """对迁移结果进行开头/结尾的平滑过渡"""
        T = rot.shape[0]
        n = min(transition_frames, T // 4)
        
        if n < 2:
            return rot
        
        # 开头: 从原始平滑过渡到迁移结果
        for i in range(n):
            alpha = i / n
            rot[i] = (1 - alpha) * original_rot[i] + alpha * rot[i]
        
        # 结尾: 从迁移结果平滑过渡到原始
        for i in range(n):
            alpha = i / n
            idx = T - 1 - i
            rot[idx] = (1 - alpha) * original_rot[idx] + alpha * rot[idx]
        
        return rot


class StyleConstraintApplicator:
    """
    风格约束应用器
    
    根据用户指定的目标风格参数, 直接修改动作旋转数据,
    产生符合敦煌舞风格约束的新序列。
    
    用法:
        applicator = StyleConstraintApplicator()
        result = applicator.apply(rotations, positions,
                                  target_arm_extension=120.0,
                                  target_spine_curvature=250.0,
                                  target_upper_lower_ratio=1.3)
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.extractor = DunhuangStyleExtractor(fps=fps)
    
    def apply(self,
              rotations: np.ndarray,
              positions: np.ndarray,
              target_arm_extension: Optional[float] = None,
              target_spine_curvature: Optional[float] = None,
              target_upper_lower_ratio: Optional[float] = None,
              target_pause_ratio: Optional[float] = None,
              target_symmetry: Optional[float] = None,
              constraint_strength: float = 0.5) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        应用风格约束, 输出修改后的旋转和位置数据
        
        Args:
            rotations: (T, J, 3) 旋转
            positions: (T, 3) 位置
            target_arm_extension: 目标上肢舒展度 (°)
            target_spine_curvature: 目标脊柱弯曲度 (°)
            target_upper_lower_ratio: 目标上下身幅度比
            target_pause_ratio: 目标停顿帧占比 (%)
            target_symmetry: 目标整体对称性 (0~1)
            constraint_strength: 约束强度 (0~1)
            
        Returns:
            (modified_rotations, modified_positions, change_report)
        """
        rot = rotations.copy()
        pos = positions.copy()
        T, J = rot.shape[:2]
        
        current = self.extractor.extract(rot, "current")
        report = {}
        
        # 记录修改前的值
        before_arm = current.left_arm_extension_mean
        before_spine = current.spine_curvature_mean
        before_ratio = current.upper_lower_ratio
        before_pause = current.pause_ratio
        before_symmetry = current.overall_symmetry
        
        # 约束1: 上肢舒展度
        if target_arm_extension is not None:
            if before_arm > 1e-3:
                ratio = target_arm_extension / before_arm
                scale = 1.0 + (ratio - 1.0) * constraint_strength
                for j in [7, 8, 9, 11, 12, 13]:
                    if j < J:
                        rot[:, j, :] *= scale
        
        # 约束2: 脊柱弯曲度
        if target_spine_curvature is not None:
            if before_spine > 1e-3:
                ratio = target_spine_curvature / before_spine
                scale = 1.0 + (ratio - 1.0) * constraint_strength
                spine_joints = [j for j in SPINE_CHAIN if j < J]
                if len(spine_joints) >= 2:
                    base = rot[:, spine_joints[0], :].copy()
                    for k in range(1, len(spine_joints)):
                        jj = spine_joints[k]
                        diff = rot[:, jj, :] - base
                        rot[:, jj, :] = base + diff * scale
        
        # 约束3: 上下身幅度比
        if target_upper_lower_ratio is not None:
            if before_ratio > 0.01 and abs(before_ratio) < 100:
                ratio_adjust = target_upper_lower_ratio / before_ratio
                scale = 1.0 + (ratio_adjust - 1.0) * constraint_strength
                upper_joints = [j for j in UPPER_BODY if j < J]
                for j in upper_joints:
                    mean_r = np.mean(rot[:, j, :], axis=0, keepdims=True)
                    rot[:, j, :] = mean_r + (rot[:, j, :] - mean_r) * scale
        
        # 约束4: 停顿节奏
        if target_pause_ratio is not None:
            rot_flat = rot.reshape(T, -1)
            vel = np.sqrt(np.sum(np.diff(rot_flat, axis=0) ** 2, axis=1))
            
            current_pause_count = int(before_pause / 100 * (T - 1))
            target_pause_count = int(target_pause_ratio / 100 * (T - 1))
            
            if target_pause_count > current_pause_count:
                extra = int((target_pause_count - current_pause_count) * constraint_strength)
                if extra > 0:
                    sorted_idx = np.argsort(vel)
                    slow_idx = sorted_idx[:current_pause_count + extra]
                    for idx in slow_idx[current_pause_count:]:
                        if 0 < idx < T - 1:
                            alpha = 0.5 * constraint_strength
                            rot[idx] = (1 - alpha) * rot[idx] + alpha * rot[idx - 1]

        # 约束5: 左右对称性
        if target_symmetry is not None:
            target_symmetry = float(np.clip(target_symmetry, 0.0, 1.0))
            current_asym = max(1e-3, 1.0 - before_symmetry)
            target_asym = max(0.0, 1.0 - target_symmetry)
            ratio_adjust = target_asym / current_asym
            diff_scale = 1.0 + (ratio_adjust - 1.0) * constraint_strength
            rot = _apply_symmetry_scale(rot, [(7, 11), (8, 12), (9, 13)], diff_scale)
            rot = _apply_symmetry_scale(rot, [(14, 18), (15, 19), (16, 20)], diff_scale)
        
        # 重新提取特征, 报告实际达成值 (而非目标值)
        after = self.extractor.extract(rot, "after")
        
        if target_arm_extension is not None:
            actual = after.left_arm_extension_mean
            report['上肢舒展度'] = f"{before_arm:.1f}° → {actual:.1f}° (目标{target_arm_extension:.0f}°, 强度{constraint_strength})"
        
        if target_spine_curvature is not None:
            actual = after.spine_curvature_mean
            report['脊柱弯曲度'] = f"{before_spine:.1f}° → {actual:.1f}° (目标{target_spine_curvature:.0f}°, 强度{constraint_strength})"
        
        if target_upper_lower_ratio is not None:
            actual = after.upper_lower_ratio
            report['上下身比例'] = f"{before_ratio:.2f} → {actual:.2f} (目标{target_upper_lower_ratio:.2f}, 强度{constraint_strength})"
        
        if target_pause_ratio is not None:
            actual = after.pause_ratio
            report['停顿比例'] = f"{before_pause:.1f}% → {actual:.1f}% (目标{target_pause_ratio:.1f}%, 强度{constraint_strength})"

        if target_symmetry is not None:
            actual = after.overall_symmetry
            report['整体对称性'] = f"{before_symmetry:.3f} → {actual:.3f} (目标{target_symmetry:.2f}, 强度{constraint_strength})"
        
        return rot, pos, report


def style_blend(rot_a: np.ndarray, rot_b: np.ndarray,
                weight: float = 0.5) -> np.ndarray:
    """
    两段动作的风格混合 (线性插值)
    
    Args:
        rot_a: (T, J, 3) 动作 A 旋转
        rot_b: (T2, J, 3) 动作 B 旋转
        weight: 混合权重 (0.0=全A, 1.0=全B)
        
    Returns:
        (T_min, J, 3) 混合后旋转
    """
    T = min(rot_a.shape[0], rot_b.shape[0])
    a = rot_a[:T]
    b = rot_b[:T]
    return (1 - weight) * a + weight * b
