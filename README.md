# 環境建置與資料準備
---

## 1. 建立 Conda 環境並安裝套件

若系統尚未安裝 `conda`，先安裝 Miniconda：
```bash
wget -c "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh" -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p "${HOME}/miniconda3"
rm -f /tmp/miniconda.sh
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
```

建立並啟用 `sfm-tts`（Python 3.10）環境，並安裝 PyTorch、訓練與評測相關套件：
```bash
conda create -n sfm-tts -c conda-forge --override-channels python=3.10 -y
conda activate sfm-tts

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
    lightning hydra-core hydra-colorlog omegaconf rootutils rich einops \
    diffusers conformer vocos torchdiffeq phonemizer unidecode "matplotlib<3.10" \
    inflect librosa scipy soundfile numba cython tensorboard tqdm gdown wget \
    transformers jiwer timm distillmos \
    "git+https://github.com/sarulab-speech/UTMOSv2.git"
```
> 若系統沒有 `libespeak-ng.so.1`（`phonemizer` 需要），請先執行：`sudo apt-get install -y espeak-ng`。

---

## 2. 下載資料集（LJ Speech / VCTK）、官方權重並更新 Filelist

### 2-1. 下載並解壓資料集
```bash
mkdir -p "${HOME}/datasets"

# LJ Speech 1.1 (~2.6 GB)
wget -c "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2" -O "${HOME}/datasets/LJSpeech-1.1.tar.bz2"
tar -xjf "${HOME}/datasets/LJSpeech-1.1.tar.bz2" -C "${HOME}/datasets"

# (VCTK) VCTK-Corpus-0.92 (~11 GB)
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate sfm-tts
conda install -c conda-forge --override-channels aria2 -y
mkdir -p "${HOME}/datasets/VCTK"
aria2c -c -x 16 -s 16 -k 1M \
    -d "${HOME}/datasets/VCTK" -o "VCTK-Corpus-0.92.zip" \
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
unzip -q -o "${HOME}/datasets/VCTK/VCTK-Corpus-0.92.zip" -d "${HOME}/datasets/VCTK"
```

### 2-2. 下載官方原版預訓練權重
請先 `cd` 進本專案根目錄: :`ShallowFlowMatching-TTS`：
```bash
cd ~/ShallowFlowMatching-TTS

mkdir -p Matcha-TTS-sfm/logs

# Matcha-TTS-sfm (LJ Speech)
wget -c "https://huggingface.co/ydqmkkx/SFM-models/resolve/main/MatchaTTS-sfm-ljspeech.ckpt" \
    -O Matcha-TTS-sfm/logs/MatchaTTS-sfm-ljspeech.ckpt

# StableTTS-sfm
wget -c "https://huggingface.co/ydqmkkx/SFM-models/resolve/main/MatchaTTS-sfm-vctk.ckpt" \
    -O Matcha-TTS-sfm/logs/MatchaTTS-sfm-vctk.ckpt
wget -c "https://huggingface.co/ydqmkkx/SFM-models/resolve/main/StableTTS-sfm-vctk.pt" \
    -O StableTTS-sfm/StableTTS-sfm-vctk.pt
```

### 2-3. 自動更新 Filelist 路徑（LJ Speech & VCTK）
在專案根目錄 `ShallowFlowMatching-TTS` 執行：
```bash
python - "$(pwd)" "${HOME}/datasets" <<'PYEOF'
import pathlib, sys

repo, data = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

# LJ Speech
ljs_wavs = data / "LJSpeech-1.1/wavs"
if ljs_wavs.exists():
    for p in (repo / "Matcha-TTS-sfm/data/LJSpeech").glob("*.txt"):
        lines = [f"{ljs_wavs / pathlib.Path(l.split('|', 1)[0]).name}|{l.split('|', 1)[1]}"
                 for l in p.read_text().splitlines() if "|" in l]
        p.write_text("\n".join(lines) + "\n")
    print("LJ Speech filelists updated.")

# VCTK (支援 wav48/*.wav 與 VCTK-0.92 *_mic1.flac)
vctk_root = data / "VCTK"
vctk_map = {}
if (vctk_root / "wav48").exists():
    vctk_map = {fp.stem: str(fp) for fp in (vctk_root / "wav48").rglob("*.wav")}
elif (vctk_root / "wav48_silence_trimmed").exists():
    for fp in (vctk_root / "wav48_silence_trimmed").rglob("*_mic1.flac"):
        vctk_map[fp.stem.replace("_mic1", "")] = str(fp)
    for fp in (vctk_root / "wav48_silence_trimmed").rglob("*_mic2.flac"):
        vctk_map.setdefault(fp.stem.replace("_mic2", ""), str(fp))

if vctk_map:
    for sub in ("Matcha-TTS-sfm/data/VCTK", "StableTTS-sfm/data/VCTK"):
        for p in (repo / sub).glob("*.txt"):
            lines = []
            for l in p.read_text().splitlines():
                if "|" not in l:
                    continue
                ap, rest = l.split("|", 1)
                stem = pathlib.Path(ap).stem.replace("_mic1", "").replace("_mic2", "")
                if stem in vctk_map:
                    lines.append(f"{vctk_map[stem]}|{rest}")
            p.write_text("\n".join(lines) + "\n")
    print("VCTK filelists updated.")
PYEOF
```
---

## 3. 執行訓練與評測


程式碼有三個部分:
1. `encoder.encoder`（文字 -> 隱藏特徵 / 粗糙頻譜）
2. `encoder.proj_sfm`（將粗糙頻譜匹配到 ODE 時間點 $t_h$）
3. `decoder.estimator` / ODE（粗糙頻譜 -> 細緻頻譜）

目前測試的是修改 `proj_sfm` 部分的 loss function 是否可以在時間與品質比原本alpha=1.0，因此載入已有權重與 fine-tune：

- 總共做 30 epoch，可在 [run_all_experiments.sh(L23)](run_all_experiments.sh#L23) 更改

- `encoder.encoder` 權重凍結
- `encoder.proj_sfm` 權重隨機化
- `ODE` 前 5 個 epoch 凍結、後 25 個 epoch fine-tune



### Matcha-TTS-sfm
- **載入權重、重置 `encoder.proj_sfm`、設定凍結範圍（只訓練 `encoder.proj_sfm` 與 `decoder`）**：
  - [Matcha-TTS-sfm/train.py(L57)](Matcha-TTS-sfm/train.py#L57)

### StableTTS-sfm
- **載入權重、重置 `encoder.sfm`、凍結 `TextEncoder`（只訓練 `encoder.sfm` 與 `decoder`）**：
  - [StableTTS-sfm/train.py(L43)](StableTTS-sfm/train.py#L43)



確認已在 `sfm-tts` 環境中（`conda activate sfm-tts`），即可背景一鍵跑完訓練與評測：

修改原論文alpha要跑的範圍(默認1~10): [eval_compare.py(L22)](eval_compare.py#L22)


一鍵跑fine tune 與 原論文比較

要跑特定實驗參考[run_all_experiments.sh(L11)](run_all_experiments.sh#L11)

```bash
bash run_all_experiments.sh 
```

輸出位置：[results_matcha_ljs.md](results_matcha_ljs.md)、[results_matcha_vctk.md](results_matcha_vctk.md)、[results_stabletts_vctk.md](results_stabletts_vctk.md)

---

## 4. 實驗結果總覽（免調 $\alpha$ vs. 原論文 $\alpha = 1.0$）

| 任務 | 系統 | $\tilde{t}_g$ | $\overline{\text{PMOS}}$ $\uparrow$ | UTMOS $\uparrow$ | UTMOSv2 $\uparrow$ | Distill-MOS $\uparrow$ | WER (%) $\downarrow$ | SIM $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Matcha + LJS** | Orig ($\alpha=1.0$) | 0.099 | 4.035 | 4.174 | 3.671 | 4.259 | 4.847 | 0.975 |
| **Matcha + LJS** | **Proposed (無 $\alpha$)** | **0.417** | **4.035** | 4.160 | 3.658 | **4.287** | **3.770** | **0.976** |
| **Matcha + VCTK** | Orig ($\alpha=1.0$) | 0.104 | 3.561 | 3.973 | 2.871 | 3.839 | 7.627 | 0.930 |
| **Matcha + VCTK** | **Proposed (無 $\alpha$)** | **0.451** | **3.679** | **4.027** | **3.056** | **3.954** | **2.260** | **0.943** |
| **StableTTS + VCTK** | Orig ($\alpha=1.0$) | 0.153 | 3.302 | 3.239 | 2.787 | 3.879 | 6.497 | 0.909 |
| **StableTTS + VCTK** | **Proposed (無 $\alpha$)** | **0.391** | **3.357** | **3.371** | **2.803** | **3.897** | **4.661** | **0.918** |
