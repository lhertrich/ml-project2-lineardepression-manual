import matplotlib.image as mpimg
import numpy as np
import random
import torch
import os


def load_image(infilename):
    """Loads the image for a given filename

    Args:
        infilename: string, the filename of the image to read

    Returns:
        data: mpimg image, the loaded data
    """
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    """Converts an image from float to uint8

    Args:
        img: float image, the image to convert

    Returns:
        rimg: uint8 image, the converted image
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def patch_to_label(patch, threshold=0.25):
    """
    Assign a label to a patch based on the average pixel intensity

    Args:
        patch: numpy array, a 16x16 patch of the predicted mask
        threshold: float, foreground threshold for binary classification

    Returns:
        int, binary label (0 or 1)
    """
    mean_value = np.mean(patch)
    if np.isnan(mean_value):
        print(f"Encountered NaN in patch mean: {patch}")
    if patch.size == 0:
        print("Encountered empty patch!")
    return 1 if mean_value > threshold else 0


def get_test_images(test_image_folder):
        """Recursively retrieve all test image file paths from the test folder

        Args:
            test_image_folder: string, the folder containing the test images

        Returns:
            image_paths: list, list of file paths to test images
        """
        image_paths = []
        for root, _, files in os.walk(test_image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths


def set_seed(seed=42):
    """ Sets the seed in numpy and torch for reproducibility

    Args:
        seed = 42: int, the seed value to use
    """
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)