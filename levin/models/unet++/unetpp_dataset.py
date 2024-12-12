import torchvision.transforms as T
import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from segmentation_models_pytorch.encoders import get_preprocessing_fn

class UnetPlusPlusRoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, mask_filenames, encoder_name, pretrained_name):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained_name)
        self.resize_transform = T.Resize((512, 512), interpolation=InterpolationMode.BICUBIC)
        self.to_tensor_transform = T.ToTensor()
        self.mask_transform = T.Compose([
            T.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Ensure image is RGB and mask is grayscale
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.resize_transform(image)
        image = self.to_tensor_transform(image)
        image = self.preprocess_input(image.permute(1, 2, 0).numpy()).transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        mask = self.mask_transform(mask)
        mask = mask.to(dtype=torch.float32)
        
        return image, mask
