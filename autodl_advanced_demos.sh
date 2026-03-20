#!/bin/bash
# ============================================================
# AutoDL 一键运行三大进阶功能
# 前提：已完成6类敦煌舞的15K训练，save/ 目录有 model000014999.pt
# ============================================================
set -e
cd /root/autodl-tmp/SinMDM
export PYTHONPATH="/root/autodl-tmp/SinMDM:$PYTHONPATH"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate SinMDM

echo "====== 进阶功能演示 开始: $(date) ======"

# ============================================================
# 步骤0: 修复 edit.py 的 BVH 保存格式（关键！）
# ============================================================
echo ">>> 修复 edit.py BVH 格式..."
sed -i 's/positions=False/positions=True/g' sample/edit.py
echo "  已将 positions=False 改为 positions=True"

# ============================================================
# 步骤1: 创建合成行走BVH（用于风格迁移）
# ============================================================
echo ""
echo ">>> 创建合成行走 BVH..."
python -c "
import sys, os, numpy as np
sys.path.insert(0, '.')
from Motion import BVH
from Motion.Animation import Animation
from Motion.Quaternions import Quaternions

def euler_to_quat(roll, pitch, yaw):
    r, p, y_ = np.radians(roll), np.radians(pitch), np.radians(yaw)
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y_/2), np.sin(y_/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])

def create_walking(template_path, output_path, n_frames, fps=30):
    anim, jn, ft = BVH.load(template_path)
    nj = len(jn)
    rots = np.zeros((n_frames, nj, 4)); rots[:,:,0] = 1.0
    pos = np.tile(anim.positions[0:1], (n_frames, 1, 1))
    t = np.arange(n_frames) / fps
    phase = 2 * np.pi * t / 0.8
    pos[:,0,2] = np.linspace(0, 40*(n_frames/fps/0.8), n_frames)
    pos[:,0,0] = anim.positions[0,0,0] + 1.5 * np.sin(phase)
    pos[:,0,1] = anim.positions[0,0,1] + 1.0 * np.abs(np.sin(phase))
    for f in range(n_frames):
        p = phase[f]
        rots[f,0] = euler_to_quat(0,0,3*np.sin(p))
        rots[f,1] = euler_to_quat(0,0,-2*np.sin(p))
        lh=25*np.sin(p); lk=40*max(0,np.sin(p)); la=10*np.sin(p+0.3)
        rh=25*np.sin(p+np.pi); rk=40*max(0,np.sin(p+np.pi)); ra=10*np.sin(p+np.pi+0.3)
        rots[f,17]=euler_to_quat(lh,0,0); rots[f,18]=euler_to_quat(-lk,0,0); rots[f,19]=euler_to_quat(la,0,0)
        rots[f,22]=euler_to_quat(rh,0,0); rots[f,23]=euler_to_quat(-rk,0,0); rots[f,24]=euler_to_quat(ra,0,0)
        rots[f,8]=euler_to_quat(15*np.sin(p+np.pi),0,0); rots[f,9]=euler_to_quat(-5,0,0)
        rots[f,13]=euler_to_quat(15*np.sin(p),0,0); rots[f,14]=euler_to_quat(-5,0,0)
        rots[f,5]=euler_to_quat(2*np.sin(p*2),0,0)
    wa = Animation(rotations=Quaternions(rots), positions=pos, orients=anim.orients, offsets=anim.offsets, parents=anim.parents)
    BVH.save(output_path, wa, jn, ft, positions=True)
    print(f'  Created {output_path}: {n_frames} frames')

frames = {'01-2-FeiTian':701,'02-1-PuSa':686,'03-2-LianHuaTongZi':855,'04-1-LiShiWuJi':345,'05-2-JiGuJiYue':318,'06-2-PiPaJiYue':599}
for name, nf in frames.items():
    tmpl = f'dataset/{name}.bvh'
    out = f'dataset/walking_{name}.bvh'
    if os.path.exists(tmpl):
        create_walking(tmpl, out, nf)
    else:
        print(f'  SKIP {name}: BVH not found')
print('  Walking BVH creation done!')
"

# ============================================================
# 步骤2: 长序列生成 (Expansion) - 04-LiShiWuJi
# 原始345帧→689帧(2倍), prefix+suffix各扩展50%
# ============================================================
echo ""
echo ">>> [1/3] 长序列生成 (Expansion) - 04-LiShiWuJi"
python -u sample/edit.py \
    --model_path save/04-1-LiShiWuJi/model000014999.pt \
    --edit_mode expansion \
    --prefix_length 0.5 \
    --suffix_length 0.5 \
    --num_samples 4 \
    --batch_size 4 \
    --output_dir save/04-1-LiShiWuJi/expansion_demo \
    --device 0
echo "<<< Expansion 完成"

# ============================================================
# 步骤3: 动作补全 (In-betweening) - 05-JiGuJiYue
# 保留前25%+后25%, 中间50%自动补全
# ============================================================
echo ""
echo ">>> [2/3] 动作补全 (In-betweening) - 05-JiGuJiYue"
python -u sample/edit.py \
    --model_path save/05-2-JiGuJiYue/model000014999.pt \
    --edit_mode in_betweening \
    --prefix_end 0.25 \
    --suffix_start 0.75 \
    --num_samples 4 \
    --batch_size 4 \
    --output_dir save/05-2-JiGuJiYue/inbetweening_demo \
    --device 0
echo "<<< In-betweening 完成"

# ============================================================
# 步骤4: 行走风格迁移 (Harmonization) - 全部6类
# 合成行走BVH + 敦煌舞模型 = 舞蹈风格的行走
# ============================================================
echo ""
echo ">>> [3/3] 行走风格迁移 (Harmonization)"
NAMES=("01-2-FeiTian" "02-1-PuSa" "03-2-LianHuaTongZi" "04-1-LiShiWuJi" "05-2-JiGuJiYue" "06-2-PiPaJiYue")

for NAME in "${NAMES[@]}"; do
    echo "  >>> Harmonization: $NAME"
    python -u sample/edit.py \
        --model_path "save/${NAME}/model000014999.pt" \
        --edit_mode harmonization \
        --ref_motion "dataset/walking_${NAME}.bvh" \
        --num_samples 2 \
        --batch_size 2 \
        --output_dir "save/${NAME}/harmonization_walking" \
        --device 0
    echo "  <<< $NAME 完成"
done

# ============================================================
# 步骤5: 验证输出
# ============================================================
echo ""
echo "====== 验证输出 ======"
echo "--- Expansion ---"
ls -la save/04-1-LiShiWuJi/expansion_demo/*.bvh 2>/dev/null || echo "  无BVH"
echo "--- In-betweening ---"
ls -la save/05-2-JiGuJiYue/inbetweening_demo/*.bvh 2>/dev/null || echo "  无BVH"
echo "--- Harmonization ---"
for NAME in "${NAMES[@]}"; do
    echo "  $NAME:"
    ls save/${NAME}/harmonization_walking/*.bvh 2>/dev/null || echo "    无BVH"
done

echo ""
echo "====== 全部进阶功能 完成: $(date) ======"
echo "请使用以下命令打包下载:"
echo "  tar czf /root/autodl-tmp/advanced_demos.tar.gz \\"
echo "    save/04-1-LiShiWuJi/expansion_demo/*.bvh \\"
echo "    save/05-2-JiGuJiYue/inbetweening_demo/*.bvh \\"
echo "    save/*/harmonization_walking/*.bvh \\"
echo "    dataset/walking_*.bvh"
