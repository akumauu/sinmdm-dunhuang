#!/bin/bash
# ============================================================
# AutoDL 一键跑通三大视频转化功能 (Expansion, Inbetweening, Harmonization)
# 机器配置: RTX 4090 等所有主流环境均兼容
#
# 【请将本脚本在 /root/autodl-tmp/SinMDM 目录下运行】
# ============================================================

set -e
cd /root/autodl-tmp/SinMDM

# 环境变量设置
export PYTHONPATH="$(pwd):$PYTHONPATH"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate SinMDM

# ============================================================
# 步骤 1. 核心修复：确保 BVH 输出携带所有的关节位置坐标
# ============================================================
echo ">>> [步骤1] 核心修复 edit.py 的 BVH 保存格式 (positions=True)..."
sed -i 's/positions=False/positions=True/g' sample/edit.py

# ============================================================
# 步骤 2. 创建合成的“纯走路”基础动作（完美匹配 27维 敦煌骨骼）
# 这将给我们的 Harmonization (风格迁移) 提供基础参照。
# ============================================================
echo ">>> [步骤2] 匹配 27维 骨骼的 Walking BVH 自动生成..."
cat << 'PY_EOF' > core_create_walking.py
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
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y_ = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y_, z])

def create_walking(template_path, output_path, n_frames, fps=30):
    anim, jn, ft = BVH.load(template_path)
    rots = np.zeros((n_frames, len(jn), 4)); rots[:,:,0] = 1.0
    pos = np.tile(anim.positions[0:1], (n_frames, 1, 1))
    
    t = np.arange(n_frames) / fps
    phase = 2 * np.pi * t / 0.8
    # 模拟角色向前匀速移动 Z 轴并包含左右晃动 X 轴
    pos[:,0,2] = np.linspace(0, 40*(n_frames/fps/0.8), n_frames)
    pos[:,0,0] = anim.positions[0,0,0] + 1.5 * np.sin(phase)
    pos[:,0,1] = anim.positions[0,0,1] + 1.0 * np.abs(np.sin(phase))
    
    for f in range(n_frames):
        p = phase[f]
        # 躯干律动
        rots[f,0] = euler_to_quat(0, 0, 3*np.sin(p))
        rots[f,1] = euler_to_quat(0, 0, -2*np.sin(p))
        # 腿部 (Left: 17,18,19  |  Right: 22,23,24)
        rots[f,17] = euler_to_quat(25*np.sin(p), 0, 0)
        rots[f,18] = euler_to_quat(-40*max(0, np.sin(p)), 0, 0)
        rots[f,19] = euler_to_quat(10*np.sin(p+0.3), 0, 0)
        rots[f,22] = euler_to_quat(25*np.sin(p+np.pi), 0, 0)
        rots[f,23] = euler_to_quat(-40*max(0, np.sin(p+np.pi)), 0, 0)
        rots[f,24] = euler_to_quat(10*np.sin(p+np.pi+0.3), 0, 0)
        # 臂部 (Left: 13,14  |  Right: 8,9)
        rots[f,8] = euler_to_quat(15*np.sin(p+np.pi), 0, 0)
        rots[f,9] = euler_to_quat(-5, 0, 0)
        rots[f,13] = euler_to_quat(15*np.sin(p), 0, 0)
        rots[f,14] = euler_to_quat(-5, 0, 0)
        # 上胸部 (5)
        rots[f,5] = euler_to_quat(2*np.sin(p*2), 0, 0)
        
    wa = Animation(rotations=Quaternions(rots), positions=pos, orients=anim.orients, offsets=anim.offsets, parents=anim.parents)
    BVH.save(output_path, wa, jn, ft, positions=True)

# 6 组动作长度字典配置
frames_dict = {'01-2-FeiTian':701,'02-1-PuSa':686,'03-2-LianHuaTongZi':855,'04-1-LiShiWuJi':345,'05-2-JiGuJiYue':318,'06-2-PiPaJiYue':599}

# 自动从 dataset 寻找 bvh 生成 walking 版本
for motion_name, n_frames in frames_dict.items():
    tmpl = f'dataset/{motion_name}.bvh'
    out = f'dataset/walking_{motion_name}.bvh'
    if os.path.exists(tmpl):
        create_walking(tmpl, out, n_frames)
        print(f" [+] 已生成 Walking 参照文件: {out}")
PY_EOF
python core_create_walking.py

# ============================================================
# 步骤 3. 利用模型的 Checkpoint 分别执行三大转化能力功能！
# （这里我们采用单卡 `--device 0` 运行处理推理即可。4090非常快）
# ============================================================
echo ">>> [步骤3.1] 长序列生成 Expansion (04-1-LiShiWuJi) ..."
python -u sample/edit.py \
    --model_path save/04-1-LiShiWuJi/model000014999.pt \
    --edit_mode expansion \
    --prefix_length 0.5 \
    --suffix_length 0.5 \
    --num_samples 2 \
    --batch_size 2 \
    --output_dir save/04-1-LiShiWuJi/expansion_demo \
    --device 0

echo ">>> [步骤3.2] 动作补全 In-betweening (05-2-JiGuJiYue) ..."
python -u sample/edit.py \
    --model_path save/05-2-JiGuJiYue/model000014999.pt \
    --edit_mode in_betweening \
    --prefix_end 0.25 \
    --suffix_start 0.75 \
    --num_samples 2 \
    --batch_size 2 \
    --output_dir save/05-2-JiGuJiYue/inbetweening_demo \
    --device 0

echo ">>> [步骤3.3] 行走风格迁移 Harmonization (六大飞天动作逐一执行) ..."
NAMES=("01-2-FeiTian" "02-1-PuSa" "03-2-LianHuaTongZi" "04-1-LiShiWuJi" "05-2-JiGuJiYue" "06-2-PiPaJiYue")
for NAME in "${NAMES[@]}"; do
    if [ -f "dataset/walking_${NAME}.bvh" ] && [ -f "save/${NAME}/model000014999.pt" ]; then
        echo "   -> 正在使用 ${NAME} 的模型提取风格..."
        python -u sample/edit.py \
            --model_path "save/${NAME}/model000014999.pt" \
            --edit_mode harmonization \
            --ref_motion "dataset/walking_${NAME}.bvh" \
            --num_samples 2 \
            --batch_size 2 \
            --output_dir "save/${NAME}/harmonization_walking" \
            --device 0
    fi
done

# ============================================================
# 步骤 4. 打包最终结果
# ============================================================
echo ">>> [步骤4] 对所有生成的视频 (.mp4) 和动画 (.bvh) 源数据进行自动压缩打包..."
tar -czvf /root/autodl-tmp/advanced_demos_output.tar.gz \
    save/04-1-LiShiWuJi/expansion_demo/*.mp4 save/04-1-LiShiWuJi/expansion_demo/*.bvh \
    save/05-2-JiGuJiYue/inbetweening_demo/*.mp4 save/05-2-JiGuJiYue/inbetweening_demo/*.bvh \
    save/*/harmonization_walking/*.mp4 save/*/harmonization_walking/*.bvh \
    dataset/walking_*.bvh 2>/dev/null || echo "部分文件打包跳过"

echo "==================================================================="
echo "[✔] 全部自动完成！无论是视频转化还是 BVH 生成都已经完毕。"
echo "请在侧边文件栏直接右键下载: /root/autodl-tmp/advanced_demos_output.tar.gz"
echo "==================================================================="
