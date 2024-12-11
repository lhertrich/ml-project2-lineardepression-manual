from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os

class UnetPlusPlusRoadSegmentationDatasetSubmission(Dataset):
    def __init__(self, image_dir, image_filenames, feature_extractor):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])

        image = Image.open(image_path).convert("RGB")  # Ensure image is RGB

        # Remove additional batch dimension added by the FeatureExtractor, because it is added by the dataloader again
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return pixel_values
