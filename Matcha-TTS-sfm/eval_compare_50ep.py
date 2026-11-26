import os
import re
import shutil
import tempfile
import warnings

import distillmos
import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
import utmosv2
from torchdiffeq import odeint
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, pipeline
from vocos import Vocos

from matcha.models.matcha_tts import MatchaTTS
from matcha.text import text_to_sequence
from matcha.utils.model import fix_len_compatibility, generate_path, sequence_mask
from matcha.utils.utils import intersperse

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_FILELIST = "data/LJSpeech/ljs_audio_text_val_filelist.txt"
ORIG_50_CKPT = "/home/zychen1204/ShallowFlowMatching-TTS-orig/Matcha-TTS-sfm/logs/ljspeech_orig_sfm/2026-11-26_03-44-13/checkpoints/last.ckpt"
PROP_50_CKPT = "/home/zychen1204/ShallowFlowMatching-TTS/Matcha-TTS-sfm/logs/ljspeech_proposed_sfm/2026-11-25_23-48-08/checkpoints/last.ckpt"
OUT_MD = "/home/zychen1204/ShallowFlowMatching-TTS/Matcha-TTS-sfm/eval_compare_50ep_results.md"
SEED = 1234

# Exact alpha values from Table 3 / Table 6 (Matcha-TTS on LJ Speech) of 2505.12226v2.pdf
PAPER_ALPHAS = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_text(text: str):
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
        dtype=torch.long,
        device=DEVICE,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=DEVICE)
    return x, x_lengths


@torch.inference_mode()
def synthesise_euler(model, x, x_lengths, n_timesteps=10, temperature=0.667, alpha=1.0):
    x_enc, align_x, logw, x_mask = model.encoder(x, x_lengths, spks=None)
    w = torch.exp(logw) * x_mask
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_max_length = y_lengths.max()
    y_max_length_ = fix_len_compatibility(y_max_length)

    y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

    x_al = torch.matmul(attn.squeeze(1).transpose(1, 2), x_enc.transpose(1, 2)).transpose(1, 2)
    mu_y, sfm_params = model.encoder.forward_smooth(x_al, y_mask)

    xp, tp, logsigma_p = sfm_params
    x0 = torch.randn_like(xp) * temperature
    sigma_p = torch.exp(logsigma_p)
    sigma_min = model.decoder.sigma_min

    Delta = torch.clamp(alpha * (torch.sqrt(sigma_p) + (1.0 - sigma_min) * tp), min=1.0)
    tp = torch.clamp(alpha * tp / Delta, max=1.0 - 1e-4)
    sigma_p = (alpha ** 2) * sigma_p / (Delta ** 2)
    xp = alpha * xp / Delta

    x_init = torch.sqrt(torch.clamp((1.0 - (1.0 - sigma_min) * tp) ** 2 - sigma_p, min=0.0)) * x0 + xp
    t_span = torch.linspace(tp.item(), 1.0, n_timesteps + 1, device=xp.device)

    def estimator(t, y):
        return model.decoder.estimator(t, y, mask=y_mask, spks=None)

    trajectory = odeint(estimator, x_init, t_span, method="euler", rtol=1e-5, atol=1e-5)
    mel = trajectory[-1][:, :, :y_max_length]
    return mel, tp.item(), torch.sqrt(sigma_p).item()


def load_gt_16k(wav_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    wav_t = torch.from_numpy(wav)
    if wav_t.ndim > 1:
        wav_t = wav_t.mean(dim=-1)
    if sr != 16000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
    return wav_t


def run_utmosv2_dir(utmosv2_model, wav_dir):
    preds = utmosv2_model.predict(input_dir=wav_dir, batch_size=16, num_workers=4, device="cuda", verbose=False)
    scores = [p["predicted_mos"] if isinstance(p, dict) else float(p) for p in preds]
    return float(np.mean(scores))


@torch.inference_mode()
def main():
    with open(VAL_FILELIST) as f:
        val_lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(val_lines)} LJ Speech validation utterances.", flush=True)

    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE).eval()
    utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEVICE).eval()
    utmosv2_model = utmosv2.create_model(pretrained=True)
    distill_model = distillmos.ConvTransformerSQAModel().to(DEVICE).eval()
    sim_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sim_model = AutoModelForAudioXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEVICE).eval()
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=0 if torch.cuda.is_available() else -1,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    def get_sim_emb(wav_16k_np):
        inputs = sim_fe(wav_16k_np, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        emb = sim_model(**inputs).embeddings
        return torch.nn.functional.normalize(emb, dim=-1)

    tmp_root = tempfile.mkdtemp(prefix="sfm_eval_50ep_")
    try:
        # 1. Ground Truth
        gt_dir = os.path.join(tmp_root, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        gt_embs, gt_texts, gt_utmos, gt_distill, gt_hyps = [], [], [], [], []
        for idx, line in enumerate(val_lines):
            wav_path, text = line.split("|", 1)
            gt_texts.append(normalize_text(text))
            w16 = load_gt_16k(wav_path)
            sf.write(os.path.join(gt_dir, f"{idx:04d}.wav"), w16.numpy(), 16000)
            w16_gpu = w16.unsqueeze(0).to(DEVICE)
            gt_utmos.append(utmos_model(w16_gpu, 16000).item())
            gt_distill.append(distill_model(w16_gpu).item())
            gt_embs.append(get_sim_emb(w16.numpy()))
            res = asr(
                {"array": w16.numpy(), "sampling_rate": 16000},
                generate_kwargs={"language": "english", "max_new_tokens": 128},
            )
            gt_hyps.append(normalize_text(res["text"]))

        gt_utmos_m = float(np.mean(gt_utmos))
        gt_utmosv2_m = run_utmosv2_dir(utmosv2_model, gt_dir)
        gt_distill_m = float(np.mean(gt_distill))
        gt_pmos_m = (gt_utmos_m + gt_utmosv2_m + gt_distill_m) / 3.0
        gt_wer = float(jiwer.wer(gt_texts, gt_hyps) * 100.0)

        print(
            "\n================ TABLE: QUALITY SWEEP (50ep vs 50ep, Euler 10 steps, Val 100 utts) ================",
            flush=True,
        )
        print(
            f"{'System':<30} | {'t_g':>5} | {'sig_g':>5} | {'PMOS ↑':>6} | {'UTMOS ↑':>7} | "
            f"{'UTMOSv2 ↑':>9} | {'Distill ↑':>9} | {'WER(%) ↓':>8} | {'SIM ↑':>5}",
            flush=True,
        )
        print("-" * 110, flush=True)
        print(
            f"{'Ground Truth':<30} | {'-':>5} | {'-':>5} | {gt_pmos_m:6.3f} | {gt_utmos_m:7.3f} | "
            f"{gt_utmosv2_m:9.3f} | {gt_distill_m:9.3f} | {gt_wer:8.3f} | {1.000:5.3f}",
            flush=True,
        )

        # Load both 50ep checkpoints
        orig_model = MatchaTTS.load_from_checkpoint(ORIG_50_CKPT, map_location=DEVICE, weights_only=False).eval()
        prop_model = MatchaTTS.load_from_checkpoint(PROP_50_CKPT, map_location=DEVICE, weights_only=False).eval()

        eval_list = [(f"Orig SFM 50ep (alpha={a:.1f})", orig_model, a) for a in PAPER_ALPHAS]
        eval_list.append(("Proposed SFM 50ep (無 alpha)", prop_model, 1.0))

        quality_rows = []
        os.makedirs("out_compare_50ep", exist_ok=True)
        for sys_idx, (name, mdl, alpha) in enumerate(eval_list):
            sys_dir = os.path.join(tmp_root, f"sys_{sys_idx}")
            os.makedirs(sys_dir, exist_ok=True)
            set_seed(SEED)
            tps, sps, utmos_l, distill_l, sims, hyps = [], [], [], [], [], []
            for idx, line in enumerate(val_lines):
                wav_path, text = line.split("|", 1)
                utt_id = os.path.splitext(os.path.basename(wav_path))[0]
                x, x_len = process_text(text)
                mel, tp, sp = synthesise_euler(mdl, x, x_len, n_timesteps=10, temperature=0.667, alpha=alpha)
                wav_24k = vocoder.decode(mel).squeeze(0).cpu()
                if idx < 3 and (alpha in (1.0, 2.5, 6.0) or "Proposed" in name):
                    tag = "proposed" if "Proposed" in name else f"orig_a{alpha:.1f}"
                    sf.write(f"out_compare_50ep/{tag}_val_line{idx+1}_{utt_id}.wav", wav_24k.numpy(), 24000, "PCM_24")
                w16 = torchaudio.functional.resample(wav_24k, 24000, 16000)
                sf.write(os.path.join(sys_dir, f"{idx:04d}.wav"), w16.numpy(), 16000)
                w16_gpu = w16.unsqueeze(0).to(DEVICE)
                tps.append(tp)
                sps.append(sp)
                utmos_l.append(utmos_model(w16_gpu, 16000).item())
                distill_l.append(distill_model(w16_gpu).item())
                emb = get_sim_emb(w16.numpy())
                sims.append(torch.sum(emb * gt_embs[idx]).item())
                res = asr(
                {"array": w16.numpy(), "sampling_rate": 16000},
                generate_kwargs={"language": "english", "max_new_tokens": 128},
            )
                hyps.append(normalize_text(res["text"]))

            u_m = float(np.mean(utmos_l))
            u2_m = run_utmosv2_dir(utmosv2_model, sys_dir)
            d_m = float(np.mean(distill_l))
            p_m = (u_m + u2_m + d_m) / 3.0
            wer_m = float(jiwer.wer(gt_texts, hyps) * 100.0)
            sim_m = float(np.mean(sims))
            shutil.rmtree(sys_dir, ignore_errors=True)

            row = {
                "name": name,
                "t_g": float(np.mean(tps)),
                "sig_g": float(np.mean(sps)),
                "pmos": p_m,
                "utmos": u_m,
                "utmosv2": u2_m,
                "distill": d_m,
                "wer": wer_m,
                "sim": sim_m,
            }
            quality_rows.append(row)
            print(
                f"{row['name']:<30} | {row['t_g']:5.3f} | {row['sig_g']:5.3f} | {row['pmos']:6.3f} | "
                f"{row['utmos']:7.3f} | {row['utmosv2']:9.3f} | {row['distill']:9.3f} | "
                f"{row['wer']:8.3f} | {row['sim']:5.3f}",
                flush=True,
            )

        # Save formatted Markdown report with ↑ / ↓ arrows
        best_pmos = max(r["pmos"] for r in quality_rows)
        best_utmos = max(r["utmos"] for r in quality_rows)
        best_utmosv2 = max(r["utmosv2"] for r in quality_rows)
        best_distill = max(r["distill"] for r in quality_rows)
        best_wer = min(r["wer"] for r in quality_rows)
        best_sim = max(r["sim"] for r in quality_rows)

        md_lines = [
            "# 50 Epochs 同條件公平對照評測結果（對照原論文 Table 3 / Table 6）",
            "",
            "- **資料集**：LJ Speech Validation Set（100 utterances）",
            "- **訓練條件**：原版與改動版皆從頭訓練 `50 epochs`、`batch_size=32`、`FP16 (16-mixed)`、`seed=1234`",
            "- **推論設定**：Euler 10 steps、`temperature=0.667`、`Vocos (24kHz)`",
            "- **評測指標**：`PMOS`（`UTMOS` / `UTMOSv2` / `Distill-MOS` 三者平均）、`WER`（`Whisper-large-v3-turbo`）、`SIM`（`WavLM-base-plus-sv`）",
            "- **箭頭說明**：`↑` 代表**越高越好**、`↓` 代表**越低越好**（模型中最佳值以粗體標示）",
            "",
            "| System | $\\tilde{t}_g$ | $\\tilde{\\sigma}_g$ | $\\overline{\\text{PMOS}}$ $\\uparrow$ | UTMOS $\\uparrow$ | UTMOSv2 $\\uparrow$ | Distill-MOS $\\uparrow$ | WER (%) $\\downarrow$ | SIM $\\uparrow$ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| Ground Truth | – | – | `{gt_pmos_m:.3f}` | `{gt_utmos_m:.3f}` | `{gt_utmosv2_m:.3f}` | `{gt_distill_m:.3f}` | `{gt_wer:.3f}` | `1.000` |",
        ]

        def fmt(val, best):
            return f"**{val:.3f}**" if abs(val - best) < 1e-6 else f"{val:.3f}"

        for r in quality_rows:
            nm = f"**{r['name']}**" if "Proposed" in r["name"] else r["name"]
            md_lines.append(
                f"| {nm} | {r['t_g']:.3f} | {r['sig_g']:.3f} | "
                f"{fmt(r['pmos'], best_pmos)} | {fmt(r['utmos'], best_utmos)} | "
                f"{fmt(r['utmosv2'], best_utmosv2)} | {fmt(r['distill'], best_distill)} | "
                f"{fmt(r['wer'], best_wer)} | {fmt(r['sim'], best_sim)} |"
            )

        with open(OUT_MD, "w") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\nSaved markdown report to: {OUT_MD}", flush=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
