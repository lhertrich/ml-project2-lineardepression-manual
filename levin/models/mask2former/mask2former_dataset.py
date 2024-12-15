import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class M2FRoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_files, mask_files, processor, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transform = transform
        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask to numpy array and preprocess
        mask = np.array(mask, dtype=np.uint8)
        mask[mask == 0] = 255
        mask[mask == 1] = 0

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        # Process with Mask2Former processor
        encoding = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),  # Remove batch dim
            "mask_labels": encoding["mask_labels"].squeeze(0),    # Binary masks
            "class_labels": encoding["class_labels"].squeeze(0)   # Class labels
        }