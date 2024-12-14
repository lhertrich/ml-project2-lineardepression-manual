import torchvision.transforms as T
import numpy as np
import os
import torch
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
from transformers import Mask2FormerImageProcessor


class M2FRoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, mask_filenames, processor_name="facebook/mask2former-swin-small-ade-semantic"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.processor = Mask2FormerImageProcessor.from_pretrained(processor_name)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Ensure Image is RGB and mask is grayscale
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Process inputs
        inputs = self.processor(
            images=image, 
            return_tensors="pt"
        )
        
        # Return preprocessed pixel values and masks without batch dimension
        pixel_values = inputs["pixel_values"].squeeze(0)  # (C, H, W)
        mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.long)
        
        return pixel_values, mask
