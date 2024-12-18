from torch.utils.data import Dataset
from PIL import Image
from transformers import SegformerImageProcessor
import os


class SegformerRoadSegmentationDatasetSubmission(Dataset):
    def __init__(self, image_dir, image_filenames):
        """ Initializes a dataset for SegFormer for submission

        Args:
            image_dir: string, the path to the image directory for the dataset
            image_filenames: list, filenames of the images for the dataset
        """
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.feature_extractor = SegformerImageProcessor(
            do_normalize=True,
            do_resize=True,
            size=512
        )

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
            image: torch.tensor, the next image as torch tensor    
        """
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])

        # Ensure image is RGB
        image = Image.open(image_path).convert("RGB")  

        # Remove additional batch dimension added by the FeatureExtractor, because it is added by the dataloader again
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return pixel_values
