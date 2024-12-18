import torchvision.transforms as T
import numpy as np
import os
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
from transformers import SegformerImageProcessor


class SegformerRoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, mask_filenames):
        """ Initializes a dataset for SegFormer for training

        Args:
            image_dir: string, the path to the image directory for the dataset
            mask_dir: string, the path to the mask directory for the dataset
            image_filenames: list, filenames of the images for the dataset
            mask_filenames: list, filenames of the masks for the dataset
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.feature_extractor = SegformerImageProcessor(
            do_normalize=True,
            do_resize=True,
            size=512
        )
        # Use nearest neighbor for masks to keep them binary
        self.mask_transform = T.Compose([
            T.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        """Calculates the length of the dataset
        
        Returns:
            int, the length of the dataset
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """ Returns an item from the dataset
        
        Args:
            idx: int, the id of the image to return

        Returns:
            pixel_values: torch.tensor, the next image as torch tensor  
            mask: torch.tensor, the next mask as torch tensor    
        """
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Ensure image is RGB and mask is grayscale
        image = Image.open(image_path).convert("RGB")  
        mask = Image.open(mask_path).convert("L")

        # Remove additional batch dimension added by the FeatureExtractor, because it is added by the dataloader again
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Transform mask
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = T.ToTensor()(mask)
        
        return pixel_values, mask
