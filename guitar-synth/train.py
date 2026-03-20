import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import AcousticGuitarPoC
from dataset import GuitarStringDataset 
import time

def envelope_loss(synth, target, frame_size=2205, hop=551):
    """Log-RMS envelope matching loss (50ms frames, 12.5ms hop at 44100Hz).
    Provides direct gradient for the KS gain parameter g when sustain is wrong.
    The STFT loss smears time (46ms+ windows) and cannot strongly penalize decay rate;
    this loss operates directly on the amplitude envelope."""
    def rms_env(x):
        frames = x.squeeze(1).unfold(-1, frame_size, hop)
        return ((frames ** 2).mean(-1) + 1e-10).sqrt() + 1e-8
    s_env, t_env = rms_env(synth), rms_env(target)
    return F.l1_loss(torch.log(s_env), torch.log(t_env))

def multi_resolution_stft_loss(synth, target):
    fft_sizes = [512, 1024, 2048]
    loss = 0.0
    for n_fft in fft_sizes:
        window = torch.hann_window(n_fft).to(synth.device)
        s_mag = torch.abs(torch.stft(synth.squeeze(1), n_fft, n_fft//4, return_complex=True, window=window)) + 1e-7
        t_mag = torch.abs(torch.stft(target.squeeze(1), n_fft, n_fft//4, return_complex=True, window=window)) + 1e-7
        
        # Spectral convergence is weighted low (0.1) to stabilize training
        sc = torch.norm(t_mag - s_mag, p="fro") / (torch.norm(t_mag, p="fro") + 1e-7)

        # Combine magnitude L1, log-magnitude L1, and spectral convergence
        loss += F.l1_loss(s_mag, t_mag) + F.l1_loss(torch.log(s_mag), torch.log(t_mag)) + (0.1 * sc)
    return loss / len(fft_sizes)

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

def run_stage(model, loader, optimizer, scheduler, device, stage_name, num_epochs):
    print(f"\n--- Starting {stage_name} ({num_epochs} Epochs) ---")
    prev_loss = None
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        
        for target, exc, pitch in loader:
            target, exc, pitch = target.to(device), exc.to(device), pitch.to(device)
            optimizer.zero_grad()
            synth, _, _, _ = model(exc, pitch)

            # Skip batches where the Karplus-Strong synthesis produced NaN/Inf values
            if torch.isnan(synth).any() or torch.isinf(synth).any():
                print("Warning: Unstable synthesis output, skipping batch.")
                continue
        
            loss = multi_resolution_stft_loss(synth, target) + 0.5 * envelope_loss(synth, target)
            loss.backward()
            
            # Clip only the parameters being optimized in this stage
            params_to_clip = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=5.0)
            optimizer.step()
            # Keep excitation_gain in a physically reasonable range
            with torch.no_grad():
                model.excitation_gain.clamp_(0.05, 3.0)
            epoch_loss += loss.item()
        
        if scheduler: scheduler.step()
        
        avg_loss = epoch_loss / len(loader)
        
        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs
        
        rel_str = " (Init)"
        if prev_loss is not None:
            diff = prev_loss - avg_loss
            percent = (diff / prev_loss) * 100
            rel_str = f" [▼ {percent:.1f}%]" if diff > 0 else f" [▲ {abs(percent):.1f}%]"
        
        print(f"[{stage_name}] Epoch {epoch+1:02d}/{num_epochs} | Loss: {avg_loss:.4f}{rel_str} | ETA: {format_time(eta)}")
        prev_loss = avg_loss

def main():
    device = torch.device("cuda")
    dataset = GuitarStringDataset("data")
    loader = DataLoader(
        dataset, 
        batch_size=8,
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    model = AcousticGuitarPoC().to(device)

    # Stage 1: Train the string model (MLP + excitation gain)
    # The body resonator is frozen so the network first learns pitch-conditioned pluck dynamics.
    model.body.requires_grad_(False)
    model.body_gains.requires_grad_(False)
    opt1 = torch.optim.Adam([
        {'params': model.mlp.parameters(), 'lr': 1e-3},
        {'params': [model.excitation_gain],  'lr': 1e-3},
    ])
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=50, eta_min=1e-5)
    run_stage(model, loader, opt1, sched1, device, "STAGE 1", 50)

    # Stage 2: Train the body resonator (IIR resonator bank gains)
    # The string model is frozen so the network focuses on shaping the body response.
    # Note: IIRResonatorBank filter coefficients are fixed buffers; only body_gains are learned.
    model.mlp.requires_grad_(False)
    model.excitation_gain.requires_grad_(False)
    model.body_gains.requires_grad_(True)
    opt2 = torch.optim.Adam([model.body_gains], lr=5e-3)
    run_stage(model, loader, opt2, None, device, "STAGE 2", 50)

    torch.save(model.state_dict(), "guitar_poc_final.pth")
    print("\nModel saved to guitar_poc_final.pth")

if __name__ == "__main__":
    main()