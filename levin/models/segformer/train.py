import os
import torchvision.transforms as T
from ...train_evaluate import TrainAndEvaluate
from ...data_preparation.dataset import RoadSegmentationDataset
from .segformer_b3 import SegFormer
from ...loss_functions import ComboLoss, DiceLoss, TverskyLoss, WeightedBCEWithLogitsLoss
from transformers import SegformerImageProcessor
from torchvision.transforms.functional import InterpolationMode
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

image_dir = os.path.abspath("data/training/augmented/images")
mask_dir = os.path.abspath("data/training/augmented/masks")

external_image_dir = os.path.abspath("data/complete_data_augmented/images")
external_mask_dir = os.path.abspath("data/complete_data_augmented/masks")

chicago_image_dir = os.path.abspath("data/complete_data_augmented_chicago/images")
chicago_mask_dir = os.path.abspath("data/complete_data_augmented_chicago/masks")

# Define transforms for images and masks
feature_extractor = SegformerImageProcessor(
    do_normalize=True,
    do_resize=True,
    size=512
)

mask_transform = T.Compose([
    T.Resize((512, 512), interpolation=InterpolationMode.NEAREST),  # Use nearest-neighbor
    T.ToTensor()
])

# Load files for original augmented, external images and chicago images
image_filenames = sorted(os.listdir(image_dir))
mask_filenames = sorted(os.listdir(mask_dir))

external_image_filenames = sorted(os.listdir(external_image_dir))
external_mask_filenames = sorted(os.listdir(external_mask_dir))

chicago_image_filenames = sorted(os.listdir(chicago_image_dir))
chicago_mask_filenames = sorted(os.listdir(chicago_mask_dir))


# Create train test and validation set for original augmented data
train_images, test_images, train_masks, test_masks = train_test_split(
    image_filenames, mask_filenames, test_size=0.2, random_state=42
)
train_images, validation_images, train_masks, validation_masks = train_test_split(
    train_images, train_masks, test_size=0.1, random_state=42)

print(f"Number of training images: {len(train_images)}")
print(f"Number of validation images: {len(validation_images)}")
print(f"Number of test images: {len(test_images)}")

# Define datasets for train and test sets
train_dataset = RoadSegmentationDataset(image_dir, mask_dir, train_images, train_masks, feature_extractor, mask_transform)
test_dataset = RoadSegmentationDataset(image_dir, mask_dir, test_images, test_masks, feature_extractor, mask_transform)
validation_dataset = RoadSegmentationDataset(image_dir, mask_dir, validation_images, validation_masks, feature_extractor, mask_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=2)


# Create train test and validation set for external data
external_train_images, external_test_images, external_train_masks, external_test_masks = train_test_split(
    external_image_filenames, external_mask_filenames, test_size=0.2, random_state=42
)
external_train_images, external_validation_images, external_train_masks, external_validation_masks = train_test_split(
    external_train_images, external_train_masks, test_size=0.1, random_state=42)

print(f"Number of external training images: {len(external_train_images)}")
print(f"Number of external validation images: {len(external_validation_images)}")
print(f"Number of external test images: {len(external_test_images)}")

# Define datasets for train and test sets
train_dataset = RoadSegmentationDataset(external_image_dir, external_mask_dir, external_train_images, external_train_masks, feature_extractor, mask_transform)
test_dataset = RoadSegmentationDataset(image_dir, mask_dir, test_images, test_masks, feature_extractor, mask_transform)
validation_dataset = RoadSegmentationDataset(image_dir, mask_dir, validation_images, validation_masks, feature_extractor, mask_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=2)


# Create train test and validation set for external chicago data
chicago_train_images, chicago_test_images, chicago_train_masks, chicago_test_masks = train_test_split(
    chicago_image_filenames, chicago_mask_filenames, test_size=0.2, random_state=42
)
chicago_train_images, chicago_validation_images, chicago_train_masks, chicago_validation_masks = train_test_split(
    chicago_train_images, chicago_train_masks, test_size=0.1, random_state=42)

print(f"Number of chicago training images: {len(chicago_train_images)}")
print(f"Number of chicago validation images: {len(chicago_validation_images)}")
print(f"Number of chicago test images: {len(chicago_test_images)}")

