import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
from diffKS import DiffKS
from core import IIRResonatorBank

class AcousticGuitarPoC(nn.Module):
    def __init__(self):
        super().__init__()
        self.diff_ks = DiffKS(
            batch_size=8,
            internal_sr=44100,
            loop_n_frames=344,
            use_double_precision=False,
            min_f0_hz=60.0
        )
        # No final Sigmoid — MLP outputs raw logits fed directly into design_loop.
        self.mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 3))

        # bias[0]=0.9 -> gain_logit=0.9 -> g=sigmoid(0.9)≈0.988 (matches real guitar decay)
        # bias[1]=0.0 -> mix_logit=0.0 -> p=sigmoid(0)=0.5   (moderate high-frequency damping)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.mlp[0].weight, gain=0.1)
            nn.init.xavier_uniform_(self.mlp[2].weight, gain=0.1)
            self.mlp[2].bias[0] = 0.9
            self.mlp[2].bias[1] = 0.0

        self.excitation_gain = nn.Parameter(torch.tensor(0.25)) # Initial gain for the excitation signal before entering the KS delay line
        self.body = IIRResonatorBank(fs=44100, min_hz=80.0, max_hz=8000.0, n_bands=24) # 24 bands (Rau & Smith, 2019)
        self.body_gains = nn.Parameter(torch.ones(1, 24) * 0.006) # Initial gain for the body resonator bank, scaled down to prevent clipping

    def forward(self, excitation, pitch):
        B = excitation.shape[0]
        mlp_out = self.mlp(pitch)

        
        loop_coeff = mlp_out[:, 0].clamp(-2.0, 2.5)  # Loop gain logit (g): controls decay time
        loop_mix   = mlp_out[:, 1].clamp(-2.5, 0.0)  # Loop mix logit (p): controls brightness/damping

        frames = excitation.shape[-1] // 256
        l_b = torch.stack([loop_coeff, loop_mix], dim=-1).unsqueeze(1).repeat(1, frames, 1)
        f0_frames = pitch.repeat(1, frames)

        # Pitch-dependent excitation pre-filter (1-pole lowpass, cutoff = 8 * f0).
        # Purpose: the KS delay line outputs raw excitation for the first loop cycle
        # (12.5ms for 80Hz) before the loop filter can attenuate high harmonics.

        exc_sq = (excitation * self.excitation_gain).squeeze(1)   # [B, N]
        N_exc = exc_sq.shape[1]
        f0_hz = pitch.squeeze(-1).clamp(min=60.0)                 # [B]

        # Pitch-conditioned low-pass pre-filter: cutoff scales with f0 (2x–8x f0)
        multiplier = (2.0 + 6.0 * (f0_hz - 60.0) / (660.0 - 60.0)).clamp(2.0, 8.0)
        cutoff_norm = (2.0 * math.pi * f0_hz * multiplier / 44100.0).clamp(max=math.pi * 0.9)
        alpha_lp = 1.0 - torch.exp(-cutoff_norm)                  # [B], in (0,1)

        # 1-pole IIR: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
        A_lp = (alpha_lp - 1.0).view(B, 1, 1).expand(B, N_exc, 1).contiguous()
        exc_filtered = sample_wise_lpc(exc_sq * alpha_lp.unsqueeze(1), A_lp)  # [B, N]

        # Run Karplus-Strong synthesis
        string_audio = self.diff_ks(f0_frames, exc_filtered, 44100, l_b)

        B_sq, N = string_audio.shape

        # Post-string low-pass filter for tonal control
        loop_post = mlp_out[:, 2].clamp(-4.0, 4.0)
        cutoff_post = torch.sigmoid(loop_post)
        cutoff_norm_post = (math.pi * cutoff_post).clamp(max=math.pi * 0.99)
        alpha_post = 1.0 - torch.exp(-cutoff_norm_post)
        A_post = (alpha_post - 1.0).view(B_sq, 1, 1).expand(B_sq, N, 1).contiguous()
        string_audio = sample_wise_lpc(string_audio * alpha_post.unsqueeze(1), A_post)

        # Body resonator bank with shared gains across the batch
        gains_expanded = self.body_gains.expand(B_sq, N, -1)

        return self.body(string_audio, gains_expanded).unsqueeze(1), loop_coeff.mean(), loop_mix.mean(), loop_post.mean()
