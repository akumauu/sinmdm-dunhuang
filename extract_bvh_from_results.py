"""Extract BVH files from edit.py results.npy output."""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Motion import BVH
from Motion.transforms import repr6d2quat
from Motion.Animation import Animation
from Motion.Quaternions import Quaternions

def extract_bvh(results_dir, original_bvh_path, repr='6d'):
    """Extract BVH files from results.npy"""
    npy_path = os.path.join(results_dir, 'results.npy')
    data = np.load(npy_path, allow_pickle=True).item()
    
    motions = data['motion']  # (n_samples, n_joints_feat, 1, n_frames)
    lengths = data['lengths']
    
    # Load original skeleton
    sin_anim, joint_names, frametime = BVH.load(original_bvh_path)
    
    joint_features_length = 9 if repr == '6d' else 7
    n_joints = motions.shape[1] // joint_features_length if motions.shape[1] % joint_features_length == 0 else sin_anim.shape[1]
    
    print(f"Motions shape: {motions.shape}")
    print(f"Lengths: {lengths}")
    print(f"Joints: {n_joints}, Features per joint: {joint_features_length}")
    
    for i in range(motions.shape[0]):
        n_frames = int(lengths[i])
        sample = motions[i]  # (n_joints_feat, 1, n_frames)
        # Reshape: (n_joints_feat, 1, n_frames) -> (n_frames, n_joints, features)
        sample = sample.transpose(2, 0, 1)[:n_frames]  # (n_frames, n_joints_feat, 1)
        sample = sample.squeeze(-1)  # (n_frames, n_joints_feat)
        sample = sample.reshape(n_frames, n_joints, joint_features_length)
        
        positions = sample[:, :, :3]
        if repr == '6d':
            quats = repr6d2quat(torch.tensor(sample[:, :, 3:])).numpy()
        else:
            quats = sample[:, :, 3:]
        
        anim = Animation(
            rotations=Quaternions(quats),
            positions=positions,
            orients=sin_anim.orients,
            offsets=sin_anim.offsets,
            parents=sin_anim.parents
        )
        
        out_path = os.path.join(results_dir, f'output_sample{i:02d}.bvh')
        BVH.save(out_path, anim, joint_names, frametime, positions=False)
        print(f"  Saved: {out_path} ({n_frames} frames)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to results directory containing results.npy")
    parser.add_argument("original_bvh", help="Path to original BVH for skeleton info")
    parser.add_argument("--repr", default="6d", choices=["6d", "quat"])
    args = parser.parse_args()
    extract_bvh(args.results_dir, args.original_bvh, args.repr)
