import os
import numpy as np
import albumentations as A
import shutil
from albumentations.core.composition import OneOf
from PIL import Image


def apply_augmentations(image, mask, num_augmentations=5):
    """Apply augmentations to an image and its mask."""
    augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        OneOf(
            [
                A.GaussNoise(var_limit=(10, 50), p=0.5),
            ],
            p=0.7,
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
    Binarizes a mask (converts roads to 255 and everything else to 0) and resizes it.
    
    Parameters:
        input_path (str): Path to the input mask image.
        output_path (str): Path to save the processed mask.
        target_size (tuple): The target size for the resized mask (width, height).
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
    image = Image.open(image_path).resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image)


def is_valid_mask(mask, road_threshold=0.5):
    """
    Check if a mask is realistic based on the percentage of red and blue pixels.
    
    Parameters:
        mask (numpy.ndarray): The mask to validate (assumes an RGB format).
        red_threshold (float): Maximum allowed proportion of red pixels (0-1).
        blue_threshold (float): Maximum allowed proportion of blue pixels (0-1).
    
    Returns:
        bool: True if the mask is valid, False otherwise.
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    
    # Extract RGB channels
    road_pixels = np.sum(mask == 255)

    road_ratio = road_pixels / total_pixels
    # Validate against threshold
    return road_ratio <= road_threshold

def process_data(existing_augmented, external_data, output_dir):
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    # Copy existing augmented data
    shutil.copytree(os.path.join(existing_augmented, 'images'), os.path.join(output_dir, 'images'), dirs_exist_ok=True)
    shutil.copytree(os.path.join(existing_augmented, 'masks'), os.path.join(output_dir, 'masks'), dirs_exist_ok=True)

    print("copied existing augmented data")

    # Process external data
    for city in os.listdir(external_data):
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
                    # Save original resized image and mask
                    #cv2.imwrite(os.path.join(output_dir, 'images', f"{base_name}_resized.png"), resized_image)
                    #cv2.imwrite(os.path.join(output_dir, 'masks', f"{base_name}_resized.png"), resized_mask)

                    # Save augmented images and masks
                    for i, (aug_image, aug_mask) in enumerate(zip(augmented_images, augmented_masks)):
                        Image.fromarray(aug_image).save(os.path.join(output_dir, 'images', f"{base_name}_aug_{i}.png"))
                        Image.fromarray(aug_mask).save(os.path.join(output_dir, 'masks', f"{base_name}_aug_{i}.png"))


def process_data_city(existing_augmented, external_data, output_dir, city):
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
output_dir = os.path.abspath("data/complete_data_augmented_chicago")
process_data_city(existing_augmented, external_data, output_dir, "chicago")