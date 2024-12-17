import os
import torchvision.transforms as T
import torch
import json
from ...helpers import set_seed
from ...train_evaluate import TrainAndEvaluate
from .deeplabv3plus_dataset import DeepLabV3PlusRoadSegmentationDataset
from .deeplabv3plus import DeepLabV3Plus
from ...loss_functions import ComboLoss, DiceLoss, TverskyLoss, WeightedBCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

set_seed()

encoder_name="resnet50"
pretrained="imagenet"

image_dir = os.path.abspath("data/training/augmented/images")
mask_dir = os.path.abspath("data/training/augmented/masks")

chicago_image_dir = os.path.abspath("data/training/chicago_data_augmented/images")
chicago_mask_dir = os.path.abspath("data/training/chicago_data_augmented/masks")

# Load files for original augmented, external images and chicago images
image_filenames = sorted(os.listdir(image_dir))
mask_filenames = sorted(os.listdir(mask_dir))

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
train_dataset = DeepLabV3PlusRoadSegmentationDataset(image_dir, mask_dir, train_images, train_masks, encoder_name=encoder_name, pretrained_name=pretrained)
test_dataset = DeepLabV3PlusRoadSegmentationDataset(image_dir, mask_dir, test_images, test_masks, encoder_name=encoder_name, pretrained_name=pretrained)
validation_dataset = DeepLabV3PlusRoadSegmentationDataset(image_dir, mask_dir, validation_images, validation_masks, encoder_name=encoder_name, pretrained_name=pretrained)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=2)


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
chicago_train_dataset = DeepLabV3PlusRoadSegmentationDataset(chicago_image_dir, chicago_mask_dir, chicago_train_images, chicago_train_masks, encoder_name=encoder_name, pretrained_name=pretrained)
chicago_test_dataset = DeepLabV3PlusRoadSegmentationDataset(chicago_image_dir, chicago_mask_dir, chicago_test_images, chicago_test_masks, encoder_name=encoder_name, pretrained_name=pretrained)
chicago_validation_dataset = DeepLabV3PlusRoadSegmentationDataset(chicago_image_dir, chicago_mask_dir, chicago_validation_images, chicago_validation_masks, encoder_name=encoder_name, pretrained_name=pretrained)

chicago_train_loader = DataLoader(chicago_train_dataset, batch_size=8, shuffle=True, num_workers=2)
chicago_test_loader = DataLoader(chicago_test_dataset, batch_size=8, shuffle=False, num_workers=2)
chicago_validation_loader = DataLoader(chicago_validation_dataset, batch_size=8, shuffle=False, num_workers=2)

# Test different loss functions for original augmented data
root_path = os.path.abspath("levin/")
save_path = os.path.join(root_path, "trained_models", "unet_plus_plus")
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
  model = DeepLabV3Plus(encoder_name=encoder_name, encoder_weight=pretrained)
  train_eval = TrainAndEvaluate(model, train_loader, validation_loader, test_loader, criterion, 10, save_path)
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

# Train and evaluate the model with the best loss function on chicago data
chicago_save_path = os.path.join(root_path, "trained_models", "segformer_b3", "chicago")
print("Created chicago save path")
os.makedirs(chicago_save_path, exist_ok=True)
model = DeepLabV3Plus
train_eval = TrainAndEvaluate(model, chicago_train_loader, chicago_validation_loader, chicago_test_loader, best_loss_function, 10, chicago_save_path)
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