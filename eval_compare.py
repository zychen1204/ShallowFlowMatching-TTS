import argparse
import os
import re
import shutil
import sys
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

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAPER_ALPHAS = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
SEED = 1234


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_gt_16k(wav_path: str) -> torch.Tensor:
    wav, sr = sf.read(wav_path, dtype="float32")
    wav_t = torch.from_numpy(wav)
    if wav_t.ndim > 1:
        wav_t = wav_t.mean(dim=-1)
    if sr != 16000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
    return wav_t


def run_utmosv2_dir(utmosv2_model, wav_dir: str) -> float:
    preds = utmosv2_model.predict(
        input_dir=wav_dir, batch_size=16, num_workers=4, device=str(DEVICE), verbose=False
    )
    return float(np.mean([float(p["predicted_mos"]) for p in preds]))


def parse_filelist(path: str, has_spk: bool, limit: int = 100):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if has_spk:
                items.append((parts[0], int(parts[1]), parts[2]))
            else:
                items.append((parts[0], None, parts[1]))
            if len(items) >= limit:
                break
    return items


# ---------- Matcha-TTS-sfm (LJ Speech & VCTK) ----------
def build_matcha_runner(repo_dir: str, orig_ckpt: str, prop_ckpt: str, is_vctk: bool):
    matcha_dir = os.path.join(repo_dir, "Matcha-TTS-sfm")
    sys.path.insert(0, matcha_dir)
    from vocos import Vocos
    from matcha.models.matcha_tts import MatchaTTS
    from matcha.text import text_to_sequence
    from matcha.utils.model import fix_len_compatibility, generate_path, sequence_mask
    from matcha.utils.utils import intersperse

    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE).eval()
    orig_model = MatchaTTS.load_from_checkpoint(orig_ckpt, map_location=DEVICE, weights_only=False).eval()
    prop_model = MatchaTTS.load_from_checkpoint(prop_ckpt, map_location=DEVICE, weights_only=False).eval()

    @torch.inference_mode()
    def synth(mdl, text, spk_id, alpha):
        x = torch.tensor(
            intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
            dtype=torch.long, device=DEVICE,
        )[None]
        x_len = torch.tensor([x.shape[-1]], dtype=torch.long, device=DEVICE)
        spks = mdl.spk_emb(torch.tensor([spk_id], dtype=torch.long, device=DEVICE)) if is_vctk else None

        x_enc, _, logw, x_mask = mdl.encoder(x, x_len, spks=spks)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max = y_lengths.max()
        y_mask = sequence_mask(y_lengths, fix_len_compatibility(y_max)).unsqueeze(1).to(x_mask.dtype)
        attn = generate_path(w_ceil.squeeze(1), (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1)).unsqueeze(1)
        x_al = torch.matmul(attn.squeeze(1).transpose(1, 2), x_enc.transpose(1, 2)).transpose(1, 2)
        _, (xp, tp, logsigma_p) = mdl.encoder.forward_smooth(x_al, y_mask)

        x0 = torch.randn_like(xp) * 0.667
        sigma_p = torch.exp(logsigma_p)
        smin = mdl.decoder.sigma_min
        Delta = torch.clamp(alpha * (torch.sqrt(sigma_p) + (1.0 - smin) * tp), min=1.0)
        tp = torch.clamp(alpha * tp / Delta, max=1.0 - 1e-4)
        sigma_p = (alpha ** 2) * sigma_p / (Delta ** 2)
        xp = alpha * xp / Delta
        x_init = torch.sqrt(torch.clamp((1.0 - (1.0 - smin) * tp) ** 2 - sigma_p, min=0.0)) * x0 + xp
        t_span = torch.linspace(tp.item(), 1.0, 11, device=xp.device)

        traj = odeint(lambda t, y: mdl.decoder.estimator(t, y, mask=y_mask, spks=spks), x_init, t_span, method="euler")
        mel = traj[-1][:, :, :y_max]
        wav_24k = vocoder.decode(mel).squeeze(0).cpu()
        return torchaudio.functional.resample(wav_24k, 24000, 16000), tp.item(), torch.sqrt(sigma_p).item()

    return orig_model, prop_model, synth


# ---------- StableTTS-sfm (VCTK) ----------
def build_stabletts_runner(repo_dir: str, orig_ckpt: str, prop_ckpt: str):
    st_dir = os.path.join(repo_dir, "StableTTS-sfm")
    sys.path.insert(0, st_dir)
    from dataclasses import asdict
    from api import get_vocoder
    from config import MelConfig, ModelConfig
    from datas.dataset import intersperse
    from models.model import StableTTS, generate_path
    from text import cleaned_text_to_sequence, symbols
    from text.english import english_to_ipa2
    from utils.mask import sequence_mask

    mel_cfg, mdl_cfg = MelConfig(), ModelConfig()
    vocoder = get_vocoder(os.path.join(st_dir, "vocoders/pretrained/vocos.pt"), "vocos").to(DEVICE).eval()

    def load_st(path):
        m = StableTTS(len(symbols), mel_cfg.n_mels, **asdict(mdl_cfg)).to(DEVICE)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        m.load_state_dict(sd, strict=True)
        return m.eval()

    orig_model, prop_model = load_st(orig_ckpt), load_st(prop_ckpt)

    @torch.inference_mode()
    def synth(mdl, text, spk_id, alpha):
        phones = english_to_ipa2(text)
        x = torch.tensor(intersperse(cleaned_text_to_sequence(phones), item=0), dtype=torch.long, device=DEVICE).unsqueeze(0)
        x_len = torch.tensor([x.size(-1)], dtype=torch.long, device=DEVICE)
        c = mdl.spk_emb(torch.tensor([spk_id], dtype=torch.long, device=DEVICE))

        x_enc, _, x_mask = mdl.encoder(x, c, x_len)
        w_ceil = torch.ceil(torch.exp(mdl.dp(x_enc, x_mask, c)) * x_mask)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max = y_lengths.max()
        y_mask = sequence_mask(y_lengths, y_max).unsqueeze(1).to(x_mask.dtype)
        attn = generate_path(w_ceil.squeeze(1), (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1)).unsqueeze(1)
        x_al = torch.matmul(attn.squeeze(1).transpose(1, 2), x_enc.transpose(1, 2)).transpose(1, 2)
        _, (xp, tp, logsigma_p) = mdl.encoder.forward_smooth(x_al, y_mask)

        sigma_p = torch.exp(logsigma_p)
        smin = mdl.decoder.sigma_min
        Delta = torch.clamp(alpha * (torch.sqrt(sigma_p) + (1.0 - smin) * tp), min=1.0)
        tp_s = torch.clamp(alpha * tp / Delta, max=1.0 - 1e-4)
        sigma_s = (alpha ** 2) * sigma_p / (Delta ** 2)
        xp_s = alpha * xp / Delta
        scaled_params = [xp_s, tp_s, torch.log(sigma_s + 1e-8)]

        cfg_kwargs = {"fake_speaker": mdl.fake_speaker, "cfg_strength": 3.0}
        mel, tp_out, sig_out = mdl.decoder(scaled_params, y_mask, 10, 1.0, c, "euler", cfg_kwargs)
        wav_44k = vocoder(mel[:, :, :y_max]).squeeze().cpu()
        return torchaudio.functional.resample(wav_44k, mel_cfg.sample_rate, 16000), float(tp_out), float(sig_out)

    return orig_model, prop_model, synth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["matcha_ljs", "matcha_vctk", "stabletts_vctk"], required=True)
    parser.add_argument("--orig-ckpt", required=True)
    parser.add_argument("--prop-ckpt", required=True)
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    has_spk = args.task != "matcha_ljs"
    items = parse_filelist(args.filelist, has_spk=has_spk, limit=args.limit)
    print(f"[{args.task}] Loaded {len(items)} validation utterances.", flush=True)

    utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEVICE).eval()
    utmosv2_model = utmosv2.create_model(pretrained=True, device=str(DEVICE))
    distill_model = distillmos.ConvTransformerSQAModel().to(DEVICE).eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0, dtype=torch.float16)
    sim_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sim_model = AutoModelForAudioXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEVICE).eval()

    @torch.inference_mode()
    def get_sim_emb(w16_np):
        inp = sim_fe(w16_np, sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
        return torch.nn.functional.normalize(sim_model(**inp).embeddings, dim=-1)

    if args.task == "stabletts_vctk":
        orig_model, prop_model, synth_fn = build_stabletts_runner(repo_dir, args.orig_ckpt, args.prop_ckpt)
    else:
        orig_model, prop_model, synth_fn = build_matcha_runner(repo_dir, args.orig_ckpt, args.prop_ckpt, is_vctk=has_spk)

    tmp_root = tempfile.mkdtemp(prefix=f"sfm_eval_{args.task}_")
    try:
        # Ground Truth
        gt_dir = os.path.join(tmp_root, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        gt_texts, gt_embs, gt_u, gt_d, gt_hyps = [], [], [], [], []
        for idx, (wav_path, _, text) in enumerate(items):
            gt_texts.append(normalize_text(text))
            w16 = load_gt_16k(wav_path)
            sf.write(os.path.join(gt_dir, f"{idx:04d}.wav"), w16.numpy(), 16000)
            w16_gpu = w16.unsqueeze(0).to(DEVICE)
            gt_u.append(utmos_model(w16_gpu, 16000).item())
            gt_d.append(distill_model(w16_gpu).item())
            gt_embs.append(get_sim_emb(w16.numpy()))
            res = asr({"array": w16.numpy(), "sampling_rate": 16000}, generate_kwargs={"language": "english"})
            gt_hyps.append(normalize_text(res["text"]))

        gt_u_m = float(np.mean(gt_u))
        gt_u2_m = run_utmosv2_dir(utmosv2_model, gt_dir)
        gt_d_m = float(np.mean(gt_d))
        gt_p_m = (gt_u_m + gt_u2_m + gt_d_m) / 3.0
        gt_wer = float(jiwer.wer(gt_texts, gt_hyps) * 100.0)

        eval_list = [(f"Orig Paper (alpha={a:.1f})", orig_model, a) for a in PAPER_ALPHAS]
        eval_list.append(("Proposed (無 alpha)", prop_model, 1.0))

        rows = []
        for s_idx, (name, mdl, alpha) in enumerate(eval_list):
            sys_dir = os.path.join(tmp_root, f"sys_{s_idx}")
            os.makedirs(sys_dir, exist_ok=True)
            set_seed(SEED)
            tps, sps, u_l, d_l, sims, hyps = [], [], [], [], [], []
            for idx, (_, spk_id, text) in enumerate(items):
                w16, tp, sp = synth_fn(mdl, text, spk_id, alpha)
                sf.write(os.path.join(sys_dir, f"{idx:04d}.wav"), w16.numpy(), 16000)
                w16_gpu = w16.unsqueeze(0).to(DEVICE)
                tps.append(tp)
                sps.append(sp)
                u_l.append(utmos_model(w16_gpu, 16000).item())
                d_l.append(distill_model(w16_gpu).item())
                sims.append(torch.sum(get_sim_emb(w16.numpy()) * gt_embs[idx]).item())
                res = asr({"array": w16.numpy(), "sampling_rate": 16000}, generate_kwargs={"language": "english"})
                hyps.append(normalize_text(res["text"]))

            u_m = float(np.mean(u_l))
            u2_m = run_utmosv2_dir(utmosv2_model, sys_dir)
            d_m = float(np.mean(d_l))
            p_m = (u_m + u2_m + d_m) / 3.0
            wer_m = float(jiwer.wer(gt_texts, hyps) * 100.0)
            sim_m = float(np.mean(sims))
            shutil.rmtree(sys_dir, ignore_errors=True)
            r = {"name": name, "t_g": float(np.mean(tps)), "sig_g": float(np.mean(sps)),
                 "pmos": p_m, "utmos": u_m, "utmosv2": u2_m, "distill": d_m, "wer": wer_m, "sim": sim_m}
            rows.append(r)
            print(f"{name:<26} | t_g={r['t_g']:.3f} | PMOS={p_m:.3f} | UTMOS={u_m:.3f} | UTMOSv2={u2_m:.3f} | Distill={d_m:.3f} | WER={wer_m:.3f}% | SIM={sim_m:.3f}", flush=True)

        best_p = max(r["pmos"] for r in rows)
        best_u = max(r["utmos"] for r in rows)
        best_u2 = max(r["utmosv2"] for r in rows)
        best_d = max(r["distill"] for r in rows)
        best_w = min(r["wer"] for r in rows)
        best_s = max(r["sim"] for r in rows)
        fmt = lambda v, b: f"**{v:.3f}**" if abs(v - b) < 1e-6 else f"{v:.3f}"

        md = [
            f"# {args.task} 評測對照表（原論文 $\\alpha=1.0\\sim10.0$ vs. 改良版無 $\\alpha$）",
            "",
            f"- 樣本數：Validation Set `{len(items)}` 句｜推論：Euler 10 steps",
            "- 箭頭說明：`↑` 越高越好、`↓` 越低越好（模型最佳值以粗體標示）",
            "",
            "| System | $\\tilde{t}_g$ | $\\tilde{\\sigma}_g$ | $\\overline{\\text{PMOS}}$ $\\uparrow$ | UTMOS $\\uparrow$ | UTMOSv2 $\\uparrow$ | Distill-MOS $\\uparrow$ | WER (%) $\\downarrow$ | SIM $\\uparrow$ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| Ground Truth | – | – | `{gt_p_m:.3f}` | `{gt_u_m:.3f}` | `{gt_u2_m:.3f}` | `{gt_d_m:.3f}` | `{gt_wer:.3f}` | `1.000` |",
        ]
        for r in rows:
            nm = f"**{r['name']}**" if "Proposed" in r["name"] else r["name"]
            md.append(
                f"| {nm} | {r['t_g']:.3f} | {r['sig_g']:.3f} | {fmt(r['pmos'], best_p)} | "
                f"{fmt(r['utmos'], best_u)} | {fmt(r['utmosv2'], best_u2)} | {fmt(r['distill'], best_d)} | "
                f"{fmt(r['wer'], best_w)} | {fmt(r['sim'], best_s)} |"
            )

        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        print(f"Saved report -> {args.out_md}", flush=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
