from torch.utils.data import Dataset
from PIL import Image
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch
import numpy as np
import os

class DeepLabV3PlusRoadSegmentationDatasetSubmission(Dataset):
    def __init__(self, image_dir, image_filenames, encoder_name, pretrained_name):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained_name)
        self.resize_transform = T.Resize((512, 512), interpolation=InterpolationMode.BICUBIC)
        self.to_tensor_transform = T.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])

        # Ensure image is RGB
        image = Image.open(image_path).convert("RGB")  

        image = self.resize_transform(image)
        image = self.to_tensor_transform(image)
        image = self.preprocess_input(image.permute(1, 2, 0).numpy()).transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        
        return image
