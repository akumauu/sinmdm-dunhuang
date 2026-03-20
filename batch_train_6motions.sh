#!/bin/bash
# ============================================================
# 六类敦煌舞 · 优化参数批量训练 (AutoDL)
# 每2000步保存checkpoint并输出BVH预览, 不生成视频
# ============================================================
set -e
cd /root/autodl-tmp/SinMDM

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate SinMDM

NAMES=("01-2-FeiTian" "02-1-PuSa" "03-2-LianHuaTongZi" "04-1-LiShiWuJi" "05-2-JiGuJiYue" "06-2-PiPaJiYue")

echo "====== 六类敦煌舞批量训练 开始: $(date) ======"

for NAME in "${NAMES[@]}"; do
  echo ""
  echo ">>> [$NAME] 开始训练 $(date)"
  python -u train/train_sinmdm.py \
    --sin_path "dataset/${NAME}.bvh" \
    --save_dir "save/${NAME}" \
    --dataset bvh_general \
    --repr 6d \
    --arch qna \
    --num_steps 15000 \
    --save_interval 2000 \
    --batch_size 64 \
    --crop_ratio 2.5 \
    --lr 0.0001 \
    --lr_method ExponentialLR \
    --lr_gamma 0.9999 \
    --gen_during_training \
    --gen_num_samples 4 \
    --overwrite
  echo "<<< [$NAME] 完成 $(date)"
done

echo "====== 全部训练完成! $(date) ======"
