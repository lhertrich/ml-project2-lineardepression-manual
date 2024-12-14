import torchvision.transforms as T
import numpy as np
import os
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor


class M2FRoadSegmentationDatasetSubmission(Dataset):
    def __init__(self, image_dir, image_filenames, processor_name="facebook/mask2former-swin-small-ade-semantic"):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.processor = AutoImageProcessor.from_pretrained(processor_name)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])

        # Ensure Image is RGB and mask is grayscale
        image = Image.open(image_path).convert("RGB")

        # Process inputs
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        
        # Return preprocessed pixel values and masks without batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values
