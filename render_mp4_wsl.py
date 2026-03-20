import sys
import os
import glob
import numpy as np

# Use the exact paths mapping in WSL
sys.path.insert(0, '/mnt/d/sinMDM/sinmdm')

from Motion import BVH
from Motion.Animation import positions_global
from Motion.AnimationStructure import get_kinematic_chain
from data_utils.humanml.utils.plot_script import plot_3d_motion

def render_bvh_to_mp4(bvh_path, output_mp4, title="Render", fps=30):
    try:
        # Load the BVH file
        anim, joint_names, frametime = BVH.load(bvh_path)
        skeleton = get_kinematic_chain(anim.parents)
        
        # Calculate globals using positions_global
        xyz_samples = positions_global(anim)  # Returns shape (n_frames, n_joints, 3)
        
        # Determine actual fps from frametime
        # frametime usually is 0.033333, leading to fps=30
        actual_fps = int(round(1.0 / frametime))
        
        print(f"Plotting {bvh_path}...")
        print(f"Shape: {xyz_samples.shape}, FPS: {actual_fps}")
        
        # Run matplotlib to render MP4
        # Note: plot_3d_motion expects motion shape: (n_frames, n_joints, 3)
        plot_3d_motion(output_mp4, skeleton, xyz_samples, dataset='bvh_general', title=title, fps=actual_fps)
        print(f"[OK] Generated {output_mp4}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to render {bvh_path}: {e}")
        return False

def main():
    base_dir = '/mnt/d/sinMDM/sinmdm/save'
    
    # Define the generated scenarios
    search_patterns = [
        # Expansion
        f"{base_dir}/04-1-LiShiWuJi/expansion_demo/sample0*.bvh",
        f"{base_dir}/04-1-LiShiWuJi/expansion_demo/input_*.bvh",
        
        # Inbetweening
        f"{base_dir}/05-2-JiGuJiYue/inbetweening_demo/sample0*.bvh",
        f"{base_dir}/05-2-JiGuJiYue/inbetweening_demo/input_*.bvh",
        
        # Harmonization
        f"{base_dir}/*/harmonization_walking/sample0*.bvh",
        f"{base_dir}/*/harmonization_walking/input_*.bvh",
    ]
    
    # Gather all BVHs
    all_bvhs = []
    for pattern in search_patterns:
        all_bvhs.extend(glob.glob(pattern))
        
    all_bvhs = sorted(list(set(all_bvhs)))
    
    if len(all_bvhs) == 0:
        print(f"No BVH files found in {base_dir}")
        return
        
    print(f"Found {len(all_bvhs)} BVH files to render.")
    
    for bvh in all_bvhs:
        # MP4 filepath
        mp4_path = bvh.replace(".bvh", ".mp4")
        
        # If input motion mapping (from edit.py)
        # Because edit.py usually saves samples and then creates rep00 mp4s, let's just make direct base mp4
        
        # Extract title from base dir name
        title = os.path.basename(os.path.dirname(bvh)) + " / " + os.path.basename(bvh).replace('.bvh','')
        
        render_bvh_to_mp4(bvh, mp4_path, title=title)

if __name__ == '__main__':
    main()
