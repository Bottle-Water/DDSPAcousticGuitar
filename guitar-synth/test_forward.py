import torch
from dataset import GuitarStringDataset
from torch.utils.data import DataLoader
from model import AcousticGuitarPoC

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load one sample from the dataset
dataset = GuitarStringDataset(data_dir="data")
dataloader = DataLoader(dataset, batch_size=1)
audio, pitch = next(iter(dataloader))

# Move data to device
audio = audio.to(device)
pitch = pitch.to(device)

print(f"Reference Audio Shape: {audio.shape}")
print(f"Target Pitch: {pitch.item()} Hz")

# 2. Initialize the model and move to device
model = AcousticGuitarPoC().to(device)

# 3. Create a fixed-length excitation burst (2s) matching training setup
target_samples = 44100 * 2
excitation = torch.zeros((1, 1, target_samples), device=device, dtype=torch.float32)
excitation[:, :, 0] = 0.5
noise_len = 882
noise = torch.randn((1, 1, noise_len), device=device, dtype=torch.float32) * 0.2
fade = torch.linspace(1.0, 0.0, noise_len, device=device).view(1, 1, -1)
excitation[:, :, 1:1+noise_len] = noise * fade

# 4. Run the forward pass
output, loop_gain, loop_mix = model(excitation, pitch)

print(f"Output Audio Shape: {output.shape}")
print(f"Has Gradients? {output.requires_grad}")

# 5. Truncate output to 1 second for a faster test
output = output[:, :, :44100]

# 6. Create a dummy loss and trigger backprop
print("Calculating Backward Pass...")
dummy_loss = output.mean()
dummy_loss.backward()

# 7. Inspect the Gradients
print("\n--- Gradient Health Check ---")
valid_grads = True
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            print(f"❌ {name}: NO GRADIENT (Disconnected Graph!)")
            valid_grads = False
        else:
            grad_norm = param.grad.norm().item()
            if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                print(f"❌ {name}: EXPLODED (NaN or Inf)")
                valid_grads = False
            else:
                print(f"✅ {name}: Grad Norm = {grad_norm:.6f}")

if valid_grads:
    print("\nSUCCESS: All gradients are connected and mathematically stable.")
else:
    print("\nFAILURE: Gradient issues detected. Check model.py for disconnected or non-differentiable operations.")