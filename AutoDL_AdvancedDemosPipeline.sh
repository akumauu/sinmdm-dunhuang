#!/bin/bash
# ============================================================
# AutoDL 一键运行三大进阶功能脚本（带 BVH 格式修复 + SDEdit 风格行走合成）
# 前提：在 /root/autodl-tmp/SinMDM 目录下执行！
# ============================================================
set -e
cd /root/autodl-tmp/SinMDM
export PYTHONPATH="/root/autodl-tmp/SinMDM:$PYTHONPATH"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate SinMDM

# ======= 1. 修复代码：确保输出带骨骼位移的正确 BVH 格式 =======
echo ">>> 修复 edit.py BVH 格式..."
sed -i 's/positions=False/positions=True/g' sample/edit.py

# ======= 2. 生成完全兼容 27维敦煌骨骼的合成行走动作BVH =======
echo ">>> 创建合成行走 BVH (用作 Reference)..."
cat << 'EOF' > create_walking_bvh.py
import sys, os, numpy as np
sys.path.insert(0, '.')
from Motion import BVH
from Motion.Animation import Animation
from Motion.Quaternions import Quaternions

def euler_to_quat(roll, pitch, yaw):
    r, p, y = np.radians(roll), np.radians(pitch), np.radians(yaw)
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    w = cr*cp*cy + sr*sp*sy; x = sr*cp*cy - cr*sp*sy; y_ = cr*sp*cy + sr*cp*sy; z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y_, z])

def create_walking(template_path, output_path, n_frames, fps=30):
    anim, jn, ft = BVH.load(template_path)
    rots = np.zeros((n_frames, len(jn), 4)); rots[:,:,0] = 1.0
    pos = np.tile(anim.positions[0:1], (n_frames, 1, 1))
    t = np.arange(n_frames) / fps; phase = 2 * np.pi * t / 0.8
    pos[:,0,2] = np.linspace(0, 40*(n_frames/fps/0.8), n_frames)
    pos[:,0,0] = anim.positions[0,0,0] + 1.5 * np.sin(phase)
    pos[:,0,1] = anim.positions[0,0,1] + 1.0 * np.abs(np.sin(phase))
    for f in range(n_frames):
        p = phase[f]
        rots[f,0] = euler_to_quat(0,0,3*np.sin(p)); rots[f,1] = euler_to_quat(0,0,-2*np.sin(p))
        rots[f,17] = euler_to_quat(25*np.sin(p),0,0); rots[f,18] = euler_to_quat(-40*max(0,np.sin(p)),0,0); rots[f,19] = euler_to_quat(10*np.sin(p+0.3),0,0)
        rots[f,22] = euler_to_quat(25*np.sin(p+np.pi),0,0); rots[f,23] = euler_to_quat(-40*max(0,np.sin(p+np.pi)),0,0); rots[f,24] = euler_to_quat(10*np.sin(p+np.pi+0.3),0,0)
        rots[f,8] = euler_to_quat(15*np.sin(p+np.pi),0,0); rots[f,9] = euler_to_quat(-5,0,0)
        rots[f,13] = euler_to_quat(15*np.sin(p),0,0); rots[f,14] = euler_to_quat(-5,0,0)
        rots[f,5] = euler_to_quat(2*np.sin(p*2),0,0)
    wa = Animation(rotations=Quaternions(rots), positions=pos, orients=anim.orients, offsets=anim.offsets, parents=anim.parents)
    BVH.save(output_path, wa, jn, ft, positions=True)

frames = {'01-2-FeiTian':701,'02-1-PuSa':686,'03-2-LianHuaTongZi':855,'04-1-LiShiWuJi':345,'05-2-JiGuJiYue':318,'06-2-PiPaJiYue':599}
for n, nf in frames.items():
    if os.path.exists(f'dataset/{n}.bvh'):
        create_walking(f'dataset/{n}.bvh', f'dataset/walking_{n}.bvh', nf)
EOF
python create_walking_bvh.py

# ======= 3. 依次跑三大模块功能 =======
echo ">>> [1/3] 执行 Expansion: 04-1-LiShiWuJi"
python -u sample/edit.py --model_path save/04-1-LiShiWuJi/model000014999.pt --edit_mode expansion --prefix_length 0.5 --suffix_length 0.5 --num_samples 2 --batch_size 2 --output_dir save/04-1-LiShiWuJi/expansion_demo

echo ">>> [2/3] 执行 In-betweening: 05-2-JiGuJiYue"
python -u sample/edit.py --model_path save/05-2-JiGuJiYue/model000014999.pt --edit_mode in_betweening --prefix_end 0.25 --suffix_start 0.75 --num_samples 2 --batch_size 2 --output_dir save/05-2-JiGuJiYue/inbetweening_demo

echo ">>> [3/3] 执行 Harmonization (行走风格迁移) - 全部六种模型"
for NAME in "01-2-FeiTian" "02-1-PuSa" "03-2-LianHuaTongZi" "04-1-LiShiWuJi" "05-2-JiGuJiYue" "06-2-PiPaJiYue"; do
    if [ -f "dataset/walking_${NAME}.bvh" ] && [ -f "save/${NAME}/model000014999.pt" ]; then
        echo "   -> 跑 Harmonization: $NAME"
        python -u sample/edit.py --model_path "save/${NAME}/model000014999.pt" --edit_mode harmonization --ref_motion "dataset/walking_${NAME}.bvh" --num_samples 2 --batch_size 2 --output_dir "save/${NAME}/harmonization_walking"
    fi
done

# ======= 4. 自动打包结果 =======
echo ">>> 全部跑完了！开始打包视频与 BVH..."
tar -czvf /root/autodl-tmp/advanced_demos_output.tar.gz \
    save/04-1-LiShiWuJi/expansion_demo/sample*.mp4 save/04-1-LiShiWuJi/expansion_demo/sample*.bvh \
    save/05-2-JiGuJiYue/inbetweening_demo/sample*.mp4 save/05-2-JiGuJiYue/inbetweening_demo/sample*.bvh \
    save/*/harmonization_walking/sample*.mp4 save/*/harmonization_walking/sample*.bvh \
    dataset/walking_*.bvh 2>/dev/null || echo "部分文件打包跳过"

echo "[OK!] 生成完毕！请下载: /root/autodl-tmp/advanced_demos_output.tar.gz"
