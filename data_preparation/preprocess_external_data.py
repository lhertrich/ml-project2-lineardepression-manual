import os
import numpy as np
import albumentations as A
import shutil
from albumentations.core.composition import OneOf
from PIL import Image


def apply_augmentations(image, mask, num_augmentations=5):
    """Apply augmentations to an image and its mask
    Args:
        image: Image, the image to augment
        maks: numpy array, the mask to augment
        num_augmentation=5: int, the number of augmentations that should be applied to image and mask

    Returns:
        augmented_images: list, a list of len(num_augmentations) that contains the augmented images
        augmented_masks: list, a list of len(num_augmentations) that contains the augmented masks
    """
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
            ],
            p=0.5,
        ),
    ]
    )

    augmented_images = []
    augmented_masks = []

    for i in range(num_augmentations):
        augmented = augmentation_pipeline(image=image, mask=mask)
        augmented_images.append(augmented['image'])
        augmented_masks.append(augmented['mask'])

    return augmented_images, augmented_masks


def binarize_and_resize_mask(input_path, target_size=(400, 400)):
    """
    Binarizes a mask (converts roads to 255 and everything else to 0) and resizes it
    
    Args:
        input_path: str, path to the input mask image
        target_size: tuple, the target size for the resized mask (width, height)

    Returns:
        numpy array, the binarized and resized mask
    """
    mask = np.array(Image.open(input_path))
    # Extract RGB channels
    rgb_mask = mask[:, :, :3] 
    # Identify road pixels (blue) and create a binary mask
    roads = (rgb_mask[:, :, 2] == 255) & (rgb_mask[:, :, 0] == 0) & (rgb_mask[:, :, 1] == 0)
    binary_mask = np.zeros_like(rgb_mask[:, :, 0], dtype=np.uint8)
    binary_mask[roads] = 255
    # Resize the mask to the target size
    binary_image = Image.fromarray(binary_mask)
    resized_mask = binary_image.resize(target_size, Image.NEAREST)
    
    return np.array(resized_mask)


def resize_image(image_path, target_size=(400, 400)):
    """Resized a given image
    
    Args:
        image_path: string, the path to the image to resize
        target_size = (400,400): tuple, the target size for resizing

    Returns:
        numpy array, the resized image
    """
    image = Image.open(image_path).resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image)


def is_valid_mask(mask, road_threshold=0.5):
    """
    Check if a mask is realistic based on the percentage of road pixels
    
    Args:
        mask: numpy array, the mask to validate (assumes an RGB format)
        roadd_threshold: float, maximum allowed proportion of road pixels (0-1)
    
    Returns:
        boolean, True if the mask is valid, False otherwise
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    
    # Extract RGB channels
    road_pixels = np.sum(mask == 255)

    road_ratio = road_pixels / total_pixels
    # Validate against threshold
    return road_ratio <= road_threshold


def process_data_city(existing_augmented, external_data, output_dir, city):
    """ Processes external data for one city

    Args:
        existing_augmented: string, the path to existing augmented images
        external_data: string, the path to the external data
        output_dir: string, the path to the directory where the processed data should be saved
        city: string, name of the subfolder where the external data for a city is stored
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    # Copy existing augmented data
    shutil.copytree(os.path.join(existing_augmented, 'images'), os.path.join(output_dir, 'images'), dirs_exist_ok=True)
    shutil.copytree(os.path.join(existing_augmented, 'masks'), os.path.join(output_dir, 'masks'), dirs_exist_ok=True)

    print("copied existing augmented data")

    # Process external data
    city_folder = os.path.join(external_data, city)
    if os.path.isdir(city_folder):
        print(f"Processing {city} images")
        for file in os.listdir(city_folder):
            if file.endswith('_image.png'):
                image_path = os.path.join(city_folder, file)
                mask_path = os.path.join(city_folder, file.replace('_image.png', '_labels.png'))

                # Resize image and mask
                resized_image = resize_image(image_path)
                resized_mask = binarize_and_resize_mask(mask_path)

                if not is_valid_mask(resized_mask):
                    print(f"Skipping invalid mask: {mask_path}")
                    continue

                # Apply augmentations
                augmented_images, augmented_masks = apply_augmentations(resized_image, resized_mask)

                # Save original and augmented data
                base_name = os.path.splitext(file)[0]

                # Save augmented images and masks
                for i, (aug_image, aug_mask) in enumerate(zip(augmented_images, augmented_masks)):
                    Image.fromarray(aug_image).save(os.path.join(output_dir, 'images', f"{base_name}_aug_{i}.png"))
                    Image.fromarray(aug_mask).save(os.path.join(output_dir, 'masks', f"{base_name}_aug_{i}.png"))


existing_augmented = os.path.abspath("data/training/augmented")
external_data = os.path.abspath("data/external_data")
output_dir_chicago = os.path.abspath("data/training/chicago_data_augmented")
process_data_city(existing_augmented, external_data, output_dir_chicago, "chicago")