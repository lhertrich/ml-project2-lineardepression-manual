from torch.utils.data import Dataset
from PIL import Image
from transformers import SegformerImageProcessor
import torchvision.transforms as T
import numpy as np
import os


class SegformerRoadSegmentationDatasetSubmission(Dataset):
    def __init__(self, image_dir, image_filenames):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.feature_extractor = SegformerImageProcessor(
            do_normalize=True,
            do_resize=True,
            size=512
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])

        image = Image.open(image_path).convert("RGB")  # Ensure image is RGB

        # Remove additional batch dimension added by the FeatureExtractor, because it is added by the dataloader again
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return pixel_values
