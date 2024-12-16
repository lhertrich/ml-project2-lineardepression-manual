import os
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoImageProcessor
from PIL import Image

class M2FImageSegmentationDataset(Dataset):
    """Image segmentation dataset using Mask2FormerImageProcessor."""

    def __init__(self, image_dir, mask_dir, image_filenames, mask_filenames, model_name="facebook/mask2former-swin-small-ade-semantic"):
        """
        Args:
            image_dir: Directory with images.
            mask_dir: Directory with masks.
            image_filenames: List of image filenames.
            mask_filenames: List of mask filenames.
            preprocessor: Mask2FormerImageProcessor.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.preprocessor = AutoImageProcessor.from_pretrained(model_name)
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
        mask = Image.open(mask_path).convert("L")  # Ensure mask is grayscale

        # Preprocess using Mask2FormerImageProcessor
        processed_inputs = self.preprocessor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        mask_labels = processed_inputs["mask_labels"][0]  # Extract tensor from list
        class_labels = processed_inputs["class_labels"][0]  # Extract tensor from list

        # Remove extra batch dimension
        processed_inputs["pixel_values"] = processed_inputs["pixel_values"].squeeze(0)
        processed_inputs["pixel_mask"] = processed_inputs["pixel_mask"].squeeze(0)

        # Update processed_inputs with unwrapped labels
        processed_inputs["mask_labels"] = mask_labels
        processed_inputs["class_labels"] = class_labels

        return processed_inputs