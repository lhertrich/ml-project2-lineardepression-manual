import albumentations as A
from albumentations.augmentations.transforms import Normalize, GaussNoise
from albumentations.core.composition import OneOf
from albumentations.augmentations import functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os

AUGMENTED_IMAGE_DIR = "data/training/augmented/images"
AUGMENTED_MASK_DIR = "data/training/augmented/masks"
IMAGE_DIR = "data/training/images"  
MASK_DIR = "data/training/groundtruth"  
PROCESS_MASK_DIR = "data/training/groundtruth_binarize"

os.makedirs(AUGMENTED_IMAGE_DIR, exist_ok=True)
os.makedirs(AUGMENTED_MASK_DIR, exist_ok=True)
os.makedirs(PROCESS_MASK_DIR, exist_ok=True)

def binarize_mask(mask, threshold=128):
    """
    Converts a non-binary mask into a binary mask using a threshold.
    
    Parameters:
        mask (numpy array): Input mask.
        threshold (int): Threshold for binarization (default: 128).
        
    Returns:
        binary_mask (numpy array): Binarized mask with values 0 and 1.
    """
    if mask.max() > 1:
        mask = mask / 255.0 
    
    binary_mask = (mask > threshold / 255.0).astype(np.uint8)
    return binary_mask


def process_masks(mask_dir, output_dir, threshold=128):
    """
    Processes all masks in a directory to ensure they are binary.
    
    Parameters:
        mask_dir (str): Path to directory containing masks.
        output_dir (str): Path to directory to save processed masks.
        threshold (int): Threshold for binarization.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    mask_files = os.listdir(mask_dir)
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path).convert('L')) 
        
        binary_mask = binarize_mask(mask, threshold)
        
        binary_mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
        binary_mask_image.save(os.path.join(output_dir, mask_file))



augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        OneOf(
            [
                A.GaussNoise(var_limit=(10, 50), p=0.5), 
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            ],
            p=0.7,
        ),
        OneOf(
            [
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
                A.ChannelShuffle(p=0.1),
            ],
            p=0.5,
        ),
    ]
)

def augment_and_save(image_dir, mask_dir, augmented_image_dir, augmented_mask_dir, num_augmentations=3):
    '''
    Augments the data and saves it
    '''
    image_files = os.listdir(image_dir)
    
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name) 
        
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        assert set(np.unique(mask)).issubset({0, 255}), f"Mask {image_name} is not binary!"
        
        for i in range(num_augmentations):

            augmented = augmentation_pipeline(image=image, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']
            
            aug_image_path = os.path.join(augmented_image_dir, f"{os.path.splitext(image_name)[0]}_aug_{i}.png")
            aug_mask_path = os.path.join(augmented_mask_dir, f"{os.path.splitext(image_name)[0]}_aug_{i}.png")
            
            Image.fromarray(augmented_image).save(aug_image_path)
            Image.fromarray(augmented_mask).save(aug_mask_path)
            
        print(f"Augmented {image_name} and saved {num_augmentations} variations.")

if __name__ == "__main__":
    
    process_masks(MASK_DIR, PROCESS_MASK_DIR, threshold=128)

    augment_and_save(
        image_dir=IMAGE_DIR,
        mask_dir=PROCESS_MASK_DIR,
        augmented_image_dir=AUGMENTED_IMAGE_DIR,
        augmented_mask_dir=AUGMENTED_MASK_DIR,
        num_augmentations=5
    )