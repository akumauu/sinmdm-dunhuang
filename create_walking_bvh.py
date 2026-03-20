"""
Create a synthetic walking motion BVH using the Dunhuang 27-joint skeleton.
This ensures perfect skeleton compatibility with the trained SinMDM models.
The walking cycle uses sinusoidal leg/arm rotations and forward root translation.
"""
import sys, os, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Motion import BVH
from Motion.Animation import Animation
from Motion.Quaternions import Quaternions


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (degrees) to quaternion [w, x, y, z]"""
    r, p, y_ = np.radians(roll), np.radians(pitch), np.radians(yaw)
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y_/2), np.sin(y_/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def create_walking_motion(template_bvh_path, output_path, n_cycles=4, fps=30):
    """
    Create a walking BVH with the same skeleton as the template.
    
    Joint indices for our 27-joint skeleton:
    0: Hips          1: Chest        2: Chest2       3: Chest3
    4: Neck          5: Head         6: Head_end     7: LeftCollar
    8: LeftUpArm     9: LeftLowArm  10: LeftHand    11: LeftHand_end
    12: RightCollar  13: RightUpArm 14: RightLowArm 15: RightHand   16: RightHand_end
    17: LeftUpLeg    18: LeftLowLeg  19: LeftFoot    20: LeftToe     21: LeftToe_end
    22: RightUpLeg   23: RightLowLeg 24: RightFoot   25: RightToe    26: RightToe_end
    """
    # Load template for skeleton structure
    anim, joint_names, frametime = BVH.load(template_bvh_path)
    print(f"Template: {len(joint_names)} joints, {anim.shape[0]} frames")
    print(f"Joints: {joint_names}")
    
    # Walking parameters
    cycle_frames = int(fps * 0.8)  # 0.8s per step cycle
    total_frames = cycle_frames * n_cycles * 2  # 2 steps per full cycle
    step_length = 40.0  # cm per step
    
    # Create identity rotations for all joints
    n_joints = len(joint_names)
    rotations = np.zeros((total_frames, n_joints, 4))
    rotations[:, :, 0] = 1.0  # w=1 for identity quaternion
    
    # Copy template positions (offsets)
    positions = np.tile(anim.positions[0:1], (total_frames, 1, 1))
    
    # Walking phase for each frame
    t = np.arange(total_frames) / fps
    phase = 2 * np.pi * t / (cycle_frames * 2 / fps)  # full cycle
    
    # === Root translation (Hips - joint 0) ===
    # Forward movement (Z-axis based on skeleton convention)
    positions[:, 0, 2] = np.linspace(0, step_length * n_cycles * 2, total_frames)
    # Lateral sway
    positions[:, 0, 0] = anim.positions[0, 0, 0] + 1.5 * np.sin(phase)
    # Vertical bounce
    positions[:, 0, 1] = anim.positions[0, 0, 1] + 1.0 * np.abs(np.sin(phase))
    
    # === Hip rotation (slight yaw oscillation) ===
    for f in range(total_frames):
        rotations[f, 0] = euler_to_quat(0, 0, 3.0 * np.sin(phase[f]))  # Hips yaw
    
    # === Spine/Chest counter-rotation ===
    for f in range(total_frames):
        rotations[f, 1] = euler_to_quat(0, 0, -2.0 * np.sin(phase[f]))  # Chest
    
    # === Leg movements ===
    hip_flex_range = 25.0   # degrees of hip flexion
    knee_flex_range = 40.0  # degrees of knee flexion
    ankle_range = 10.0
    
    for f in range(total_frames):
        p = phase[f]
        
        # Left leg (phase = p)
        left_hip_angle = hip_flex_range * np.sin(p)
        left_knee_angle = knee_flex_range * max(0, np.sin(p))  # only flex, no hyperextend
        left_ankle_angle = ankle_range * np.sin(p + 0.3)
        
        # Right leg (phase = p + pi, opposite)
        right_hip_angle = hip_flex_range * np.sin(p + np.pi)
        right_knee_angle = knee_flex_range * max(0, np.sin(p + np.pi))
        right_ankle_angle = ankle_range * np.sin(p + np.pi + 0.3)
        
        # Left leg joints
        rotations[f, 17] = euler_to_quat(left_hip_angle, 0, 0)    # LeftUpLeg
        rotations[f, 18] = euler_to_quat(-left_knee_angle, 0, 0)  # LeftLowLeg (negative = flex)
        rotations[f, 19] = euler_to_quat(left_ankle_angle, 0, 0)  # LeftFoot
        
        # Right leg joints
        rotations[f, 22] = euler_to_quat(right_hip_angle, 0, 0)   # RightUpLeg
        rotations[f, 23] = euler_to_quat(-right_knee_angle, 0, 0) # RightLowLeg
        rotations[f, 24] = euler_to_quat(right_ankle_angle, 0, 0) # RightFoot
        
        # === Arm swing (opposite to legs) ===
        arm_swing = 15.0
        left_arm_angle = arm_swing * np.sin(p + np.pi)   # opposite to left leg
        right_arm_angle = arm_swing * np.sin(p)           # opposite to right leg
        
        rotations[f, 8] = euler_to_quat(left_arm_angle, 0, 0)   # LeftUpArm
        rotations[f, 9] = euler_to_quat(-5, 0, 0)                # LeftLowArm (slight bend)
        rotations[f, 13] = euler_to_quat(right_arm_angle, 0, 0) # RightUpArm
        rotations[f, 14] = euler_to_quat(-5, 0, 0)              # RightLowArm
        
        # === Head slight bob ===
        rotations[f, 5] = euler_to_quat(2 * np.sin(phase[f] * 2), 0, 0)  # Head nod at double freq
    
    # Create animation
    walk_anim = Animation(
        rotations=Quaternions(rotations),
        positions=positions,
        orients=anim.orients,
        offsets=anim.offsets,
        parents=anim.parents
    )
    
    # Save BVH
    BVH.save(output_path, walk_anim, joint_names, frametime, positions=False)
    print(f"Saved: {output_path} ({total_frames} frames, {total_frames/fps:.1f}s)")
    return output_path


if __name__ == "__main__":
    # Use LiShiWuJi as template (best model)
    template = r"D:\sinMDM\sinmdm\dataset\04-1-LiShiWuJi.bvh"
    output = r"D:\sinMDM\sinmdm\dataset\synthetic_walking.bvh"
    create_walking_motion(template, output, n_cycles=4, fps=30)
    
    # Also create one matching JiGuJiYue's skeleton
    template2 = r"D:\sinMDM\sinmdm\dataset\05-2-JiGuJiYue.bvh"
    output2 = r"D:\sinMDM\sinmdm\dataset\synthetic_walking_jigu.bvh"
    create_walking_motion(template2, output2, n_cycles=4, fps=30)
    
    print("\nDone! Walking BVH files created.")
