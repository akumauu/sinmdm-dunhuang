#!/bin/bash
# ============================================================
# 六类敦煌舞 · 批量生成预览 (AutoDL 版)
# ============================================================
set -e
cd /root/autodl-tmp/SinMDM

NAMES=("01-2-FeiTian" "02-1-PuSa" "03-2-LianHuaTongZi" "04-1-LiShiWuJi" "05-2-JiGuJiYue" "06-2-PiPaJiYue")

echo "====== 批量生成 开始: $(date) ======"

for NAME in "${NAMES[@]}"; do
  CKPT=""
  for step in 12500 15000 14999 10000 7500 5000; do
    f=$(printf "save/%s/model%09d.pt" "$NAME" "$step")
    [ -f "$f" ] && CKPT="$f" && break
  done
  [ -z "$CKPT" ] && CKPT=$(ls -t "save/${NAME}"/model*.pt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && echo "⚠️ ${NAME}: 无checkpoint, 跳过" && continue

  echo ">>> [$NAME] 生成中 ckpt=${CKPT}"
  python -u sample/generate.py --model_path "${CKPT}" --num_samples 10 --seed 10
  echo "<<< [$NAME] 完成"
done

echo "====== 全部生成完成! $(date) ======"
