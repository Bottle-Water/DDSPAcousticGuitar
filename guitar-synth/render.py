import os
import math
import torch
import torchaudio
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram
from dataset import GuitarStringDataset
from model import AcousticGuitarPoC
from torchlpc import sample_wise_lpc

sys.path.append(os.path.abspath("diffKS_torchLPC"))

SAMPLE_RATE = 44100

VARIANTS = [
    ("full",        "Full model (KS + filtered excitation + post LPF + body resonators)"),
    ("ks_filtered", "KS + filtered excitation + post LPF (no body resonators)"),
    ("ks_raw",      "KS only (no excitation filter, no post LPF, no body resonators)"),
]


def normalize(audio, peak=0.9):
    p = audio.abs().max()
    if p > 0.01:
        return audio / p * peak
    return audio


def save_wav(path, audio_tensor, sr):
    """audio_tensor: [1, N] or [N]"""
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    torchaudio.save(path, audio_tensor.cpu(), sr)


def save_spectrogram(audio_np, sr, path, title):
    """Save a linear-frequency STFT spectrogram (0-8 kHz, dB scale) as a PNG."""
    f, t, Sxx = scipy_spectrogram(
        audio_np, fs=sr,
        nperseg=2048, noverlap=1792,
        scaling='spectrum'
    )
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

    freq_mask = f <= 8000
    f = f[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]

    vmax = Sxx_db.max()
    vmin = vmax - 80.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno',
                  vmin=vmin, vmax=vmax)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    plt.colorbar(ax.collections[0], ax=ax, label='dB')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_spectrogram(audios_np, titles, sr, path, note_label):
    """4-panel spectrogram comparison: target + 3 pipeline stages."""
    fig, axes = plt.subplots(len(audios_np), 1, figsize=(12, 3.5 * len(audios_np)),
                              sharex=True)
    if len(audios_np) == 1:
        axes = [axes]

    for ax, audio, title in zip(axes, audios_np, titles):
        f, t, Sxx = scipy_spectrogram(
            audio, fs=sr,
            nperseg=2048, noverlap=1792,
            scaling='spectrum'
        )
        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
        freq_mask = f <= 8000
        f = f[freq_mask]
        Sxx_db = Sxx_db[freq_mask, :]
        vmax = Sxx_db.max()
        vmin = vmax - 80.0
        im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno',
                           vmin=vmin, vmax=vmax)
        ax.set_ylabel('Freq (Hz)')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label='dB')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Pipeline ablation — {note_label}', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_ablations(model, exc_batch, pitch_batch, device):
    """
    Run the full forward pass and return all three pipeline-stage outputs.

    Returns:
        full_out        [B, 1, N]  KS (filtered exc) + post LPF + body resonators
        ks_filtered     [B, 1, N]  KS (filtered exc) + post LPF, no body
        ks_raw          [B, 1, N]  KS (raw exc), no post LPF, no body
        lc_mean, lm_mean, lp_mean: scalar loop-param stats
    """
    B = exc_batch.shape[0]

    # --- Loop parameters ---
    mlp_out    = model.mlp(pitch_batch)
    loop_coeff = mlp_out[:, 0].clamp(-2.0, 2.5)
    loop_mix   = mlp_out[:, 1].clamp(-3.0, 0.0)
    loop_post  = mlp_out[:, 2].clamp(-4.0, 4.0)

    frames    = exc_batch.shape[-1] // 256
    l_b       = torch.stack([loop_coeff, loop_mix], dim=-1).unsqueeze(1).repeat(1, frames, 1)
    f0_frames = pitch_batch.repeat(1, frames)

    # --- Raw excitation (gain only, no pre-filter) ---
    exc_raw = (exc_batch * model.excitation_gain).squeeze(1)   # [B, N]

    # --- Filtered excitation (pitch-dependent pre-filter) ---
    f0_hz       = pitch_batch.squeeze(-1).clamp(min=60.0)      # [B]
    multiplier = (2.0 + 6.0 * (f0_hz - 60.0) / (660.0 - 60.0)).clamp(2.0, 8.0)
    cutoff_norm = (2.0 * math.pi * f0_hz * multiplier / 44100.0).clamp(max=math.pi * 0.9)
    alpha_lp    = 1.0 - torch.exp(-cutoff_norm)
    N_exc       = exc_raw.shape[1]
    A_lp        = (alpha_lp - 1.0).view(B, 1, 1).expand(B, N_exc, 1).contiguous()
    exc_filtered = sample_wise_lpc(exc_raw * alpha_lp.unsqueeze(1), A_lp)
    exc_filtered = sample_wise_lpc(exc_filtered * alpha_lp.unsqueeze(1), A_lp)
    # --- DiffKS passes ---
    string_raw      = model.diff_ks(f0_frames, exc_raw,      44100, l_b)  # [B, N]
    string_filtered = model.diff_ks(f0_frames, exc_filtered, 44100, l_b)  # [B, N]

    B_sq, N = string_filtered.shape

    # --- Post-string LPF (applied to filtered path only) ---
    cutoff_post      = torch.sigmoid(loop_post)
    cutoff_norm_post = (math.pi * cutoff_post).clamp(max=math.pi * 0.99)
    alpha_post       = 1.0 - torch.exp(-cutoff_norm_post)             # [B]
    A_post           = (alpha_post - 1.0).view(B_sq, 1, 1).expand(B_sq, N, 1).contiguous()
    string_filtered  = sample_wise_lpc(string_filtered * alpha_post.unsqueeze(1), A_post)

    # --- Body resonators (full model only) ---
    gains_exp = model.body_gains.expand(B_sq, N, -1)
    full      = model.body(string_filtered, gains_exp).unsqueeze(1)    # [B, 1, N]

    ks_raw_out      = string_raw.unsqueeze(1)       # [B, 1, N]
    ks_filtered_out = string_filtered.unsqueeze(1)  # [B, 1, N]

    return full, ks_filtered_out, ks_raw_out, loop_coeff.mean(), loop_mix.mean(), loop_post.mean()


def main():
    # 1. Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load model
    model = AcousticGuitarPoC().to(device)
    checkpoint_path = 'guitar_poc_final.pth'

    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # 3. Dataset - pick low / mid / high notes
    dataset = GuitarStringDataset(data_dir="data", target_sample_rate=SAMPLE_RATE)
    all_notes = sorted([(i, dataset[i][2].item()) for i in range(len(dataset))], key=lambda x: x[1])
    selected_indices = [all_notes[0][0], all_notes[len(all_notes) // 2][0], all_notes[-1][0]]
    labels = ["low", "mid", "high"]

    # 4. Results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 5. Build padded batch (DiffKS requires exactly batch_size=8)
    BATCH = model.diff_ks.batch_size  # 8
    targets, excitations, pitches, hzs = [], [], [], []
    for idx in selected_indices:
        t, e, p = dataset[idx]
        targets.append(t); excitations.append(e); pitches.append(p); hzs.append(p.item())

    n   = len(selected_indices)
    pad = BATCH - n
    exc_batch   = torch.stack(excitations + [excitations[-1]] * pad).to(device)  # [8, 1, N]
    pitch_batch = torch.stack(pitches     + [pitches[-1]]     * pad).to(device)  # [8, 1]

    # 6. Run ablation forward passes
    with torch.no_grad():
        full_out, ks_filtered_out, ks_raw_out, lc_mean, lm_mean, lp_mean = \
            compute_ablations(model, exc_batch, pitch_batch, device)

    variant_tensors = {
        "full":        full_out[:n],
        "ks_filtered": ks_filtered_out[:n],
        "ks_raw":      ks_raw_out[:n],
    }

    # 7. Save audio + spectrograms per note
    for i, label in enumerate(labels):
        hz     = hzs[i]
        hz_tag = f"{label}_{int(hz)}Hz"
        print(f"\n[{label}] {hz:.2f} Hz  "
              f"(gain_logit={lc_mean.item():.3f}  "
              f"mix_logit={lm_mean.item():.3f}  "
              f"post_logit={lp_mean.item():.3f})")

        # --- Target ---
        target_audio = targets[i].cpu()
        target_np    = target_audio.squeeze(0).numpy()
        target_wav   = os.path.join(results_dir, f"{hz_tag}_target.wav")
        target_spec  = os.path.join(results_dir, f"{hz_tag}_target_spec.png")
        save_wav(target_wav, target_audio, SAMPLE_RATE)
        save_spectrogram(target_np, SAMPLE_RATE, target_spec, f"Target — {hz_tag}")
        print(f"  Saved: {target_wav}")
        print(f"  Saved: {target_spec}")

        # --- Pipeline variants ---
        panel_audios = [target_np]
        panel_titles = [f"Target — {hz_tag}"]

        for key, desc in VARIANTS:
            audio_t   = normalize(variant_tensors[key][i]).cpu()
            audio_np  = audio_t.squeeze(0).numpy()
            wav_path  = os.path.join(results_dir, f"{hz_tag}_{key}.wav")
            spec_path = os.path.join(results_dir, f"{hz_tag}_{key}_spec.png")
            save_wav(wav_path, audio_t, SAMPLE_RATE)
            save_spectrogram(audio_np, SAMPLE_RATE, spec_path, f"{desc}\n{hz_tag}")
            print(f"  Saved: {wav_path}")
            print(f"  Saved: {spec_path}")
            panel_audios.append(audio_np)
            panel_titles.append(desc)

        # --- Comparison panel ---
        comp_path = os.path.join(results_dir, f"{hz_tag}_comparison.png")
        save_comparison_spectrogram(panel_audios, panel_titles, SAMPLE_RATE,
                                    comp_path, f"{label} ({hz:.1f} Hz)")
        print(f"  Saved: {comp_path}")

    # 8. Print body gains
    print("\n--- Trained Body Gains ---")
    gains = model.body_gains.detach().cpu().squeeze().tolist()
    for i, g in enumerate(gains):
        print(f"  Band {i+1:2d}: {g:.6f}")

    # 9. Print per-note MLP outputs for diagnostics
    print("\n--- Per-note MLP outputs ---")
    with torch.no_grad():
        for idx, label in zip(selected_indices, labels):
            _, _, p = dataset[idx]
            p_dev = p.unsqueeze(0).to(device)
            out = model.mlp(p_dev)
            gc = out[0, 0].clamp(-2.0, 2.5)
            gm = out[0, 1].clamp(-3.0, 0.0)
            gp = out[0, 2].clamp(-4.0, 4.0)
            g     = torch.sigmoid(gc)
            p_val = torch.sigmoid(gm)
            lp_cut = torch.sigmoid(gp)
            print(f"  [{label}] {p.item():.1f}Hz | "
                  f"gain_logit={gc:.3f} g={g:.4f} | "
                  f"mix_logit={gm:.3f} p={p_val:.3f} | "
                  f"post_logit={gp:.3f} cutoff_frac={lp_cut:.3f}")

    # 10. Synthesize arbitrary pitch
    test_hz = 440.0  # change this to test any frequency
    BATCH = model.diff_ks.batch_size
    N = int(SAMPLE_RATE * 2.0)
    _, exc, _ = dataset[selected_indices[-1]]  # use high note excitation
    excitation = exc.unsqueeze(0).repeat(BATCH, 1, 1).to(device)
    pitch = torch.full((BATCH, 1), test_hz).to(device)
    with torch.no_grad():
        synth, _, _, _ = model(excitation, pitch)
    audio = normalize(synth[0].cpu())
    out_path = os.path.join(results_dir, f"interp_{int(test_hz)}Hz.wav")
    save_wav(out_path, audio, SAMPLE_RATE)
    spec_path = os.path.join(results_dir, f"interp_{int(test_hz)}Hz_spec.png")
    save_spectrogram(audio.squeeze(0).numpy(), SAMPLE_RATE, spec_path, f"Interpolated — {test_hz}Hz")
    print(f"\nSaved interpolated note: {out_path}")


if __name__ == "__main__":
    main()