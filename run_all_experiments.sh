#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Shallow Flow Matching TTS 訓練與評測腳本（請先依 SETUP.md 建置環境與資料）
# 流程：載入官方權重 + 凍結 TextEncoder + 重置 SFM Head + 微調 ODE -> 自動評測
#   1. Matcha-TTS-sfm + LJ Speech  -> results_matcha_ljs.md
#   2. Matcha-TTS-sfm + VCTK       -> results_matcha_vctk.md
#   3. StableTTS-sfm  + VCTK       -> results_stabletts_vctk.md
#
# 若想單獨跑其中一項，可用環境變數控制（1=執行, 0=跳過）：
#   1. 只跑 Matcha-TTS-sfm + LJ Speech：
#      RUN_LJS=1 RUN_VCTK_MATCHA=0 RUN_VCTK_STABLETTS=0 bash run_all_experiments.sh
#   2. 只跑 Matcha-TTS-sfm + VCTK：
#      RUN_LJS=0 RUN_VCTK_MATCHA=1 RUN_VCTK_STABLETTS=0 bash run_all_experiments.sh
#   3. 只跑 StableTTS-sfm + VCTK：
#      RUN_LJS=0 RUN_VCTK_MATCHA=0 RUN_VCTK_STABLETTS=1 bash run_all_experiments.sh
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BATCH_SIZE=32
EPOCHS=30          
WARMUP_EPOCHS=5 # 5 epoch 暖身 + 25 epoch 微調 ODE

RUN_LJS="${RUN_LJS:-1}"
RUN_VCTK_MATCHA="${RUN_VCTK_MATCHA:-1}"
RUN_VCTK_STABLETTS="${RUN_VCTK_STABLETTS:-1}"

# ---------- 1. Matcha-TTS-sfm + LJ Speech ----------
if [ "${RUN_LJS}" = "1" ]; then
    cd "${REPO_DIR}/Matcha-TTS-sfm"
    PRETRAINED_CKPT="logs/MatchaTTS-sfm-ljspeech.ckpt" SFM_WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
    python -u train.py \
        experiment=ljspeech \
        data.train_filelist_path=data/LJSpeech/ljs_audio_text_train_filelist.txt \
        data.batch_size="${BATCH_SIZE}" \
        data.num_workers=0 \
        model.optimizer.lr=1e-4 \
        trainer.precision=16-mixed \
        trainer.max_epochs="${EPOCHS}" \
        run_name=ljspeech_sfm_finetune

    cd "${REPO_DIR}"
    LJS_PROP_CKPT="$(ls -t Matcha-TTS-sfm/logs/ljspeech_sfm_finetune/*/checkpoints/last.ckpt | head -n 1)"
    python -u eval_compare.py \
        --task matcha_ljs \
        --orig-ckpt Matcha-TTS-sfm/logs/MatchaTTS-sfm-ljspeech.ckpt \
        --prop-ckpt "${LJS_PROP_CKPT}" \
        --filelist Matcha-TTS-sfm/data/LJSpeech/ljs_audio_text_val_filelist.txt \
        --out-md results_matcha_ljs.md
fi

# ---------- 2. Matcha-TTS-sfm + VCTK ----------
if [ "${RUN_VCTK_MATCHA}" = "1" ]; then
    cd "${REPO_DIR}/Matcha-TTS-sfm"
    PRETRAINED_CKPT="logs/MatchaTTS-sfm-vctk.ckpt" SFM_WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
    python -u train.py \
        experiment=vctk \
        data.batch_size="${BATCH_SIZE}" \
        data.num_workers=0 \
        model.optimizer.lr=1e-4 \
        trainer.precision=16-mixed \
        trainer.max_epochs="${EPOCHS}" \
        run_name=vctk_sfm_finetune

    cd "${REPO_DIR}"
    VCTK_PROP_CKPT="$(ls -t Matcha-TTS-sfm/logs/vctk_sfm_finetune/*/checkpoints/last.ckpt | head -n 1)"
    python -u eval_compare.py \
        --task matcha_vctk \
        --orig-ckpt Matcha-TTS-sfm/logs/MatchaTTS-sfm-vctk.ckpt \
        --prop-ckpt "${VCTK_PROP_CKPT}" \
        --filelist Matcha-TTS-sfm/data/VCTK/vctk_audio_sid_text_val_filelist.txt \
        --out-md results_matcha_vctk.md
fi

# ---------- 3. StableTTS-sfm + VCTK ----------
if [ "${RUN_VCTK_STABLETTS}" = "1" ]; then
    cd "${REPO_DIR}/StableTTS-sfm"
    [ ! -f "filelists/vctk.json" ] && python -u preprocess.py
    sed -i -E "s/^(\s*batch_size:\s*int\s*=\s*)[0-9]+/\1${BATCH_SIZE}/" config.py
    sed -i -E "s/^(\s*num_epochs:\s*int\s*=\s*)[0-9]+/\1${EPOCHS}/" config.py
    sed -i -E "s/^(\s*save_interval_epoch:\s*int\s*=\s*)[0-9]+/\1${EPOCHS}/" config.py
    PRETRAINED_CKPT="StableTTS-sfm-vctk.pt" SFM_WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
    python -u train.py

    cd "${REPO_DIR}"
    ST_PROP_CKPT="$(ls -t StableTTS-sfm/checkpoints/checkpoint_*.pt | head -n 1)"
    python -u eval_compare.py \
        --task stabletts_vctk \
        --orig-ckpt StableTTS-sfm/StableTTS-sfm-vctk.pt \
        --prop-ckpt "${ST_PROP_CKPT}" \
        --filelist StableTTS-sfm/data/VCTK/vctk_audio_sid_text_val_filelist.txt \
        --out-md results_stabletts_vctk.md
fi

echo "全部指定實驗完成！"
