#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.multiprocessing
from datasets import load_dataset
from PIL import Image
from torchvision.utils import save_image
from diffusers import AutoencoderDC
torch.multiprocessing.set_sharing_strategy('file_system')

ds = load_dataset("evanarlian/imagenet_1k_resized_256")

device = torch.device("cuda")

dc_ae: AutoencoderDC = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", torch_dtype=torch.float32).to(device).eval()

# ========= Custom Dataset Wrapper =========
class CustomImageDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data  # expects ds['train']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Convert PIL image to RGB and apply transform
        image = item['image'].convert("RGB")
        x = self.transform(image)
        label = item['label']
        return {'id': idx, 'x': x, 'label': label}

# -------------------------------
# Transformation for the Autoencoder
# -------------------------------
transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

# Create dataset and DataLoader
train_dataset = CustomImageDataset(ds['train'], transform)
batch_size = 32  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# -------------------------------
# Fixed Quantization Parameters
# -------------------------------
min_val, max_val = -6.5, 6.5
range_val = max_val - min_val

# List to store encoded results
encoded_results = []

# Put your autoencoder model into evaluation mode
dc_ae.eval()
with torch.no_grad():
    for batch in tqdm(train_loader, desc="Encoding dataset"):
        ids = batch['id']
        labels = batch['label']
        x_batch = batch['x'].to(device)
        # Encode the images
        latent = dc_ae.encode(x_batch).latent  # expected shape: [batch, C, H, W]
        # Quantize the latent representation to 8-bit
        latent_clipped = torch.clamp(latent, min_val, max_val)
        latent_normalized = (latent_clipped - min_val) / range_val  # scale to [0,1]
        latent_uint8 = (latent_normalized * 255).round().to(torch.uint8)
        # Save each sample's id, quantized latent, and label
        for i in range(latent_uint8.size(0)):
            encoded_results.append({
                'id': ids[i],
                'latent': latent_uint8[i].cpu(),
                'label': labels[i]
            })

# Save the encoded dataset
torch.save(encoded_results, "imagenet_3gb.pt")
print("Saved encoded dataset to imagenet_3gb.pt")
