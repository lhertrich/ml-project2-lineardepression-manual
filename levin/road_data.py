from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os

class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, gt_dir, transform, patch_size=512):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.files = os.listdir(image_dir)
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load image and ground truth
        img_path = f"{self.image_dir}/{self.files[idx]}"
        gt_path = f"{self.gt_dir}/{self.files[idx]}"
        
        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        # Resize both to patch_size
        img = img.resize((self.patch_size, self.patch_size))
        gt = gt.resize((self.patch_size, self.patch_size))

        # Apply any transformations (e.g., normalization)
        img = np.array(img) / 255.0
        gt = np.array(gt) / 255.0

        # Convert to PyTorch tensors
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        gt = torch.tensor(gt, dtype=torch.float).unsqueeze(0)

        return img, gt
