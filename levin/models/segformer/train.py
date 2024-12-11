import os
import torchvision.transforms as T
import torch
import json
from ...helpers import set_seed
from ...train_evaluate import TrainAndEvaluate
from segformer_dataset import RoadSegmentationDataset
from .segformer_b3 import SegFormer
from ...loss_functions import ComboLoss, DiceLoss, TverskyLoss, WeightedBCEWithLogitsLoss
from transformers import SegformerImageProcessor
from torchvision.transforms.functional import InterpolationMode
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

set_seed()

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

### Original augmented data
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


### Complete augmented original and external data
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
external_train_dataset = RoadSegmentationDataset(external_image_dir, external_mask_dir, external_train_images, external_train_masks, feature_extractor, mask_transform)
external_test_dataset = RoadSegmentationDataset(external_image_dir, external_mask_dir, external_test_images, external_test_masks, feature_extractor, mask_transform)
external_validation_dataset = RoadSegmentationDataset(external_image_dir, external_mask_dir, external_validation_images, external_validation_masks, feature_extractor, mask_transform)

external_train_loader = DataLoader(external_train_dataset, batch_size=8, shuffle=True, num_workers=2)
external_test_loader = DataLoader(external_test_dataset, batch_size=8, shuffle=False, num_workers=2)
external_validation_loader = DataLoader(external_validation_dataset, batch_size=8, shuffle=False, num_workers=2)


### Augmented original and chicago data
# Create train test and validation set for external chicago data
chicago_train_images, chicago_test_images, chicago_train_masks, chicago_test_masks = train_test_split(
    chicago_image_filenames, chicago_mask_filenames, test_size=0.2, random_state=42
)
chicago_train_images, chicago_validation_images, chicago_train_masks, chicago_validation_masks = train_test_split(
    chicago_train_images, chicago_train_masks, test_size=0.1, random_state=42)

print(f"Number of chicago training images: {len(chicago_train_images)}")
print(f"Number of chicago validation images: {len(chicago_validation_images)}")
print(f"Number of chicago test images: {len(chicago_test_images)}")

# Define datasets for train and test sets
chicago_train_dataset = RoadSegmentationDataset(chicago_image_dir, chicago_mask_dir, chicago_train_images, chicago_train_masks, feature_extractor, mask_transform)
chicago_test_dataset = RoadSegmentationDataset(chicago_image_dir, chicago_mask_dir, chicago_test_images, chicago_test_masks, feature_extractor, mask_transform)
chicago_validation_dataset = RoadSegmentationDataset(chicago_image_dir, chicago_mask_dir, chicago_validation_images, external_validation_masks, feature_extractor, mask_transform)

chicago_train_loader = DataLoader(chicago_train_dataset, batch_size=8, shuffle=True, num_workers=2)
chicago_test_loader = DataLoader(chicago_test_dataset, batch_size=8, shuffle=False, num_workers=2)
chicago_validation_loader = DataLoader(chicago_validation_dataset, batch_size=8, shuffle=False, num_workers=2)

# Test different loss functions for original augmented data
root_path = os.path.abspath("levin/")
save_path = os.path.join(root_path, "trained_models", "segformer_b3")
print("Created save path")
os.makedirs(save_path, exist_ok=True)

loss_functions = [torch.nn.BCEWithLogitsLoss(), WeightedBCEWithLogitsLoss(), DiceLoss(), ComboLoss(), TverskyLoss()]
best_f1 = 0
best_threshold = 0
best_epoch = 0
best_loss_function = None
best_data = "original"
for criterion in loss_functions:
  print(f"Training model with {criterion.__class__.__name__}")
  # Train and evaluate
  model = SegFormer()
  train_eval = TrainAndEvaluate(model, train_loader, validation_loader, test_loader, criterion, 10, save_path, save=False)
  f1, threshold, epoch = train_eval.run()
  if f1 > best_f1:
    best_f1 = f1
    best_threshold = threshold
    best_epoch = epoch
    best_loss_function = criterion

print(f"Best F1 score: {best_f1}")
print(f"Best threshold: {best_threshold}")
print(f"Best epoch: {best_epoch}")
print(f"Best loss function: {best_loss_function.__class__.__name__}")


# Train and evaluate the model with the best loss function on external data
external_save_path = os.path.join(root_path, "trained_models", "segformer_b3", "external")
print("Created external save path")
os.makedirs(external_save_path, exist_ok=True)
model = SegFormer()
train_eval = TrainAndEvaluate(model, external_train_loader, external_validation_loader, external_test_loader, best_loss_function, 10, external_save_path)
f1, threshold, epoch = train_eval.run()
if f1 > best_f1:
    best_f1 = f1
    best_threshold = threshold
    best_epoch = epoch
    best_data = "external"


# Train and evaluate the model with the best loss function on chicago data
chicago_save_path = os.path.join(root_path, "trained_models", "segformer_b3", "chicago")
print("Created chicago save path")
os.makedirs(chicago_save_path, exist_ok=True)
model = SegFormer()
train_eval = TrainAndEvaluate(model, external_train_loader, external_validation_loader, external_test_loader, best_loss_function, 10, chicago_save_path)
f1, threshold, epoch = train_eval.run()
if f1 > best_f1:
    best_f1 = f1
    best_threshold = threshold
    best_epoch = epoch
    best_data = "chicago"


results = {"best_criterion": best_loss_function,
           "best_f1": best_f1,
           "best_threshold": best_threshold,
           "best_epoch": best_epoch,
           "best_data": best_data
           }

results_path = os.path.join(root_path, "models", "segformer", "train_results.json")
with open(results_path, "w") as json_file:
   json.dump(results, json_file, indent=4)

print(f"Best data: {best_data}, best_criterion: {best_loss_function} with f1: {best_f1}, best epoch: {best_epoch}, best_threshold: {best_threshold}")
print("Finished training, everything saved")