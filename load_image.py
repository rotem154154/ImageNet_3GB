#!/usr/bin/env python3
import torch
from torchvision.utils import save_image
from diffusers import AutoencoderDC


device = torch.device("cuda")

dc_ae: AutoencoderDC = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", torch_dtype=torch.float32).to(device).eval()

# Also ensure that your device is defined:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the encoded dataset
encoded_data = torch.load("imagenet_3gb.pt")

# Select a sample to decode (for instance, the first sample)
sample = encoded_data[0]
latent_uint8 = sample['latent']  # This is the 8-bit quantized latent

# -------------------------------
# Fixed Quantization Parameters
# -------------------------------
min_val, max_val = -6.5, 6.5
range_val = max_val - min_val

# Dequantize the latent back to float32
latent_dequantized = latent_uint8.float() / 255.0 * range_val + min_val
latent_dequantized = latent_dequantized.unsqueeze(0).to(device)

# Decode the latent using the autoencoder
dc_ae.eval()
with torch.no_grad():
    decoded = dc_ae.decode(latent_dequantized).sample

# Prepare the decoded image for saving.
# Assuming the decoded image is normalized to [-0.5, 0.5], convert to [0,1]
decoded_img = decoded.squeeze(0) * 0.5 + 0.5
decoded_img = decoded_img.clamp(0, 1)

# Save the image
save_image(decoded_img, "decoded_sample.png")
print("Saved decoded image to decoded_sample.png")
