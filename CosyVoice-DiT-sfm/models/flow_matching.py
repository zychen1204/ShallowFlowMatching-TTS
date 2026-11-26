import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchdiffeq import odeint

from models.estimator import Decoder

# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFMDecoder(torch.nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 1e-4

        self.estimator = Decoder(noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, p_dropout, n_layers, n_heads, kernel_size, gin_channels)

    @torch.inference_mode()
    def forward(self, sfm_params, mask, n_timesteps, temperature=1.0, c=None, solver=None, cfg_kwargs=None):
        xp, tp, logsigma_p = sfm_params
        x0 = torch.randn_like(xp) * temperature
        sigma_p = torch.exp(logsigma_p)

        Delta = torch.clamp(torch.sqrt(sigma_p) + (1.0 - self.sigma_min) * tp, min=1.0)
        tp = torch.clamp(tp / Delta, max=1.0 - 1e-4)
        sigma_p = sigma_p / (Delta ** 2)
        xp = xp / Delta

        x = torch.sqrt(torch.clamp((1.0 - (1.0 - self.sigma_min) * tp) ** 2 - sigma_p, min=0.0)) * x0 + xp
        t_span = torch.linspace(tp.item(), 1.0, n_timesteps + 1, device=xp.device)
        
        # cfg control
        if cfg_kwargs is None:
            estimator = functools.partial(self.estimator, mask=mask, c=c)
        else:
            estimator = functools.partial(self.cfg_wrapper, mask=mask, c=c, cfg_kwargs=cfg_kwargs)
            
        trajectory = odeint(estimator, x, t_span, method=solver, rtol=1e-5, atol=1e-5)
        return trajectory[-1], tp.item(), torch.sqrt(sigma_p).item()
    
    # cfg inference
    def cfg_wrapper(self, t, x, mask, c, cfg_kwargs):
        fake_speaker = cfg_kwargs['fake_speaker'].repeat(x.size(0), 1)
        cfg_strength = cfg_kwargs['cfg_strength']
        
        cond_output = self.estimator(t, x, mask, c)
        uncond_output = self.estimator(t, x, mask, fake_speaker)
        
        output = uncond_output + cfg_strength * (cond_output - uncond_output)
        return output

    def compute_loss(self, x1, mask, sfm_params, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            sfm_params: list of [xp, tp, logsigma_p]
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss_dict, value_dict
        """
        b, n_feats, _ = x1.shape
        xp, tp, logsigma_p = sfm_params
        kappa = 1.0 - self.sigma_min

        # 1. Exact masked Frobenius orthogonal projection in fp32
        x1_f = (x1 * mask).float()
        xp_d = (xp.detach() * mask).float()
        denom_bins = torch.clamp(torch.sum(mask.float(), dim=[1, 2], keepdim=True) * n_feats, min=1.0)

        p1 = torch.sum(x1_f * x1_f, dim=[1, 2], keepdim=True) / denom_bins + 1e-8
        ip = torch.sum(xp_d * x1_f, dim=[1, 2], keepdim=True) / denom_bins
        xp_norm2 = torch.sum(xp_d * xp_d, dim=[1, 2], keepdim=True) / denom_bins + 1e-8

        th_raw = ip / p1
        th_pos = torch.clamp(th_raw, min=0.0)
        sigma_h_f = torch.sum(((xp_d - th_pos * x1_f) * mask.float()) ** 2, dim=[1, 2], keepdim=True) / denom_bins

        # 2. Variational information-optimal CondOT target scale c* (singularity-free)
        rho_h = torch.where(
            ip > 0.0,
            torch.clamp((ip * ip) / (xp_norm2 * p1), min=0.0, max=1.0 - 1e-6),
            torch.zeros_like(ip),
        )
        q_err = (1.0 - rho_h) * p1
        c_star_f = (2.0 * kappa) / (
            2.0 * (kappa ** 2) * rho_h
            + q_err
            + torch.sqrt(q_err * (q_err + 8.0 * (kappa ** 2) * rho_h) + 1e-12)
        )
        tau_star_f = rho_h * c_star_f
        c_star = c_star_f.to(x1.dtype)

        # De-biased mean loss: eliminates contraction collapse and aligns th -> tau*
        mu_loss = F.mse_loss(xp * mask, c_star * x1 * mask, reduction="sum") / (torch.sum(mask) * n_feats)

        # 3. Theorem 1 CondOT state construction
        Delta_f = torch.clamp(torch.sqrt(sigma_h_f + 1e-12) + kappa * th_pos, min=1.0)
        th_f = torch.clamp(th_pos / Delta_f, max=1.0 - 1e-4)
        sigma_h_scaled_f = sigma_h_f / (Delta_f ** 2)

        Delta = Delta_f.to(x1.dtype)
        th = th_f.to(x1.dtype)
        sigma_h = sigma_h_scaled_f.to(x1.dtype)
        xp = (xp / Delta) * mask

        t_loss = F.mse_loss(tp.float(), th_f, reduction="mean")
        sigma_loss = F.mse_loss(logsigma_p.float(), torch.log(sigma_h_scaled_f + 1e-8), reduction="mean")

        # 4. Tangent-space gradient projection (bit-exact forward identity, blocks radial scale drift)
        xp_sg = xp.detach()
        xp_sg_f = xp_sg.float()
        rad_proj = torch.sum((xp - xp_sg).float() * xp_sg_f, dim=[1, 2], keepdim=True) / (
            torch.sum(xp_sg_f * xp_sg_f, dim=[1, 2], keepdim=True) + 1e-8
        )
        xp_cfm = xp - rad_proj.to(xp.dtype) * xp_sg

        # 5. Single-segment piecewise flow on [th, 1] with cosine timestep scheduler
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        t = 1.0 - torch.cos(t * 0.5 * torch.pi)

        x0 = torch.randn_like(x1)
        x_th = torch.sqrt(torch.clamp((1.0 - kappa * th) ** 2 - sigma_h, min=0.0)) * x0 + xp_cfm
        xt = ((1.0 - t) * x_th + t * (x1 + self.sigma_min * x0)) * mask
        ut = ((x1 + self.sigma_min * x0 - x_th) / (1.0 - th)) * mask
        t = (1.0 - th) * t + th

        vt = self.estimator(t.squeeze(), xt, mask, c)
        fm_loss = F.mse_loss(vt, ut, reduction="sum") / (torch.sum(mask) * ut.size(1))
        return {
            "fm_loss": fm_loss,
            "mu_loss": mu_loss,
            "t_loss": t_loss,
            "sigma_loss": sigma_loss,
        }, {
            "sigma_h": torch.mean(sigma_h_scaled_f),
            "sigma_p": torch.mean(torch.exp(logsigma_p.float())),
            "th": torch.mean(th_f),
            "tp": torch.mean(tp.float()),
            "tau_star": torch.mean(tau_star_f),
            "Delta": torch.mean(Delta_f),
        }
