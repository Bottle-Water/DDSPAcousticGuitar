import os
import re
import torch
import torchaudio
from torch.utils.data import Dataset

class GuitarStringDataset(Dataset):
    def __init__(self, data_dir="data", target_sample_rate=44100):
        if not os.path.exists(data_dir) and os.path.exists(f"../{data_dir}"):
            data_dir = f"../{data_dir}"
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.target_samples = 88200
        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith(('.wav', '.WAV'))]

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(os.path.join(self.data_dir, self.file_list[idx]))
        
        # 1. Resample to target rate and convert to mono
        if sr != self.target_sample_rate: 
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 2. Trim leading silence/latency
        # We find the first sample that exceeds 5% of the absolute peak
        threshold = 0.05 * torch.max(torch.abs(waveform))
        onsets = (torch.abs(waveform) > threshold).nonzero()
        
        if len(onsets) > 0:
            first_sample = onsets[0, 1]
            waveform = waveform[:, first_sample:] # Slice to start exactly at the pluck
            
        # 3. Peak Normalization (0.95)
        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform)) * 0.95

        # 4. Crop length for uniformity (pad if too short)
        if waveform.shape[-1] > self.target_samples:
            waveform = waveform[..., :self.target_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_samples - waveform.shape[-1]))

        # 5. Pitch Detection
        match = re.search(r'(?i)([0-9.]+)hz', self.file_list[idx])
        pitch = torch.tensor([float(match.group(1)) if match else 80.0])

        # 6. Excitation - Starts at sample 0 to match trimmed audio
        # Seed by idx so each file gets the same noise burst every epoch.
        excitation = torch.zeros_like(waveform)
        pluck_len = int(self.target_sample_rate * 0.01) # 10ms
        rng = torch.Generator().manual_seed(idx * 2654435761)
        excitation[..., :pluck_len] = torch.rand(1, pluck_len, generator=rng) * 2.0 - 1.0
        excitation[..., 0] = 1.0 # Ensure the very first sample is a strong impulse to trigger the string
        
        return waveform, excitation, pitch