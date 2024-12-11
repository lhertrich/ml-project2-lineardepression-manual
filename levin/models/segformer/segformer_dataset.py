import torchvision.transforms as T
import numpy as np
import os
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
from transformers import SegformerImageProcessor


class SegformerRoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, mask_filenames):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.feature_extractor = feature_extractor = SegformerImageProcessor(
            do_normalize=True,
            do_resize=True,
            size=512
        )
        self.mask_transform = mask_transform = T.Compose([
            T.Resize((512, 512), interpolation=InterpolationMode.NEAREST),  # Use nearest-neighbor
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
        mask = Image.open(mask_path).convert("L")  # Ensure mask is grayscale

        # Remove additional batch dimension added by the FeatureExtractor, because it is added by the dataloader again
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = T.ToTensor()(mask)
        
        return pixel_values, mask
