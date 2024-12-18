import torchvision.transforms as T
import os
import torch
from torch.utils.data import Dataset
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from transformers import SegformerImageProcessor
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class CombinedRoadSegmentationDatasetSubmission(Dataset):
    def __init__(self, image_dir, image_filenames, encoder_name, pretrained_name):
        """ Initializes a dataset for CombinedModel for submission

        Args:
            image_dir: string, the path to the image directory for the dataset
            image_filenames: list, filenames of the images for the dataset
            encoder_name: string, name of the used encoder for the corresponding model
            pretrained_name: string, name of the dataset which was chosen for the pretrained weights
        """
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained_name)
        self.resize_transform = T.Resize((512, 512), interpolation=InterpolationMode.BICUBIC)
        self.segformer_feature_extractor = SegformerImageProcessor(
            do_normalize=True,
            do_resize=True,
            size=512
        )
        self.to_tensor_transform = T.ToTensor()


    def __len__(self):
        """Calculates the length of the dataset
        
        Returns:
            int, the length of the dataset
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """ Returns the next two images for the combined models from the dataset
        
        Args:
            idx: int, the id of the two images to return

        Returns:
            deeplab_image: torch.tensor, the next image for the DeepLabV3+ model as torch tensor
            pixel_values: torch.tensor, the next image for the SegFormer model as torch tensor     
        """
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])

        # Ensure image is RGB
        image = Image.open(image_path).convert("RGB")  

        # Process the image for DeepLabV3+
        deeplab_image = self.resize_transform(image)
        deeplab_image = self.to_tensor_transform(deeplab_image)
        deeplab_image = self.preprocess_input(deeplab_image.permute(1, 2, 0).numpy()).transpose(2, 0, 1)
        deeplab_image = torch.tensor(deeplab_image, dtype=torch.float32)

        # Process the image for Segformer
        pixel_values = self.segformer_feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return deeplab_image, pixel_values