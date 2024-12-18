import os
import torch
from utils.helpers import set_seed
from models.deep_lab_V3_plus.deeplabv3plus_dataset import DeepLabV3PlusRoadSegmentationDataset
from models.deep_lab_V3_plus.deeplabv3plus import DeepLabV3Plus
from models.segformer.segformer_b3 import SegFormer
from models.combined_model.combined_model import CombinedModels
from models.combined_model.combined_model_dataset_submission import CombinedRoadSegmentationDatasetSubmission
from utils.create_submission import CreateSubmission
from utils.helpers import get_test_images
from loss_functions import ComboLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Set random seed for reproducability
set_seed()

# Define parameters for DeepLabV3+
encoder_name_chicago="resnet101"
encoder_weight_chicago="imagenet"

# Load chicago training data to train both models
chicago_image_dir = os.path.abspath("data/training/chicago_data_augmented/images")
chicago_mask_dir = os.path.abspath("data/training/chicago_data_augmented/masks")
chicago_image_filenames = sorted(os.listdir(chicago_image_dir))
chicago_mask_filenames = sorted(os.listdir(chicago_mask_dir))

# Create validation split for training (not useful, overlaps with training data)
chicago_train_images, chicago_validation_images, chicago_train_masks, chicago_validation_masks = train_test_split(
    chicago_image_filenames, chicago_mask_filenames, test_size=0.05, random_state=42
)

print(f"Number of chicago submission images: {len(chicago_image_filenames)}")
print(f"Number of chicago validation images: {len(chicago_validation_images)}")


### Train DeepLabV3+ model

# Create datasets and dataloaders for DeepLabV3+, train on the full training data
submission_dataset = DeepLabV3PlusRoadSegmentationDataset(chicago_image_dir, chicago_mask_dir, chicago_image_filenames, chicago_mask_filenames, encoder_name_chicago, encoder_weight_chicago)
chicago_validation_dataset = DeepLabV3PlusRoadSegmentationDataset(chicago_image_dir, chicago_mask_dir, chicago_validation_images, chicago_validation_masks, encoder_name_chicago, encoder_weight_chicago)
submission_loader = DataLoader(submission_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
chicago_submission_validation_loader = DataLoader(chicago_validation_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

# Train model
loss_function = ComboLoss()
root_path = os.path.abspath("trained_models")
chicago_save_path = os.path.join(root_path, "trained_models", "combined_models", "submission")
os.makedirs(chicago_save_path, exist_ok=True)
print("Created deeplab save path")
deeplab_model = DeepLabV3Plus(encoder_name=encoder_name_chicago, encoder_weight=encoder_weight_chicago, in_channels=3, num_labels=1)
deeplab_model.train(submission_loader, chicago_submission_validation_loader, loss_function, chicago_save_path, epochs=10, save=False)


### Train SegFormer model

# Create datasets and dataloaders for SegFormer, train on the full training data
submission_loader = DataLoader(submission_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
chicago_submission_validation_loader = DataLoader(chicago_validation_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
print(f"Number of chicago submission images: {len(chicago_image_filenames)}")
print(f"Number of chicago validation images: {len(chicago_validation_images)}")

# Train model
loss_function = torch.nn.BCEWithLogitsLoss()
seg_model = SegFormer()
seg_model.train(submission_loader, chicago_submission_validation_loader, loss_function, chicago_save_path, epochs=8, save=False)


### Combine models and create submission
combined_model = CombinedModels(deeplab_model, seg_model)

# Load test data and create dataloader for submission
test_image_dir = os.path.abspath("data/test_set_images")
test_images = get_test_images(test_image_dir)
test_images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
test_dataset = CombinedRoadSegmentationDatasetSubmission(test_image_dir, test_images, encoder_name_chicago, encoder_weight_chicago)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f"Loaded {len(test_images)} test images and created dataloader")

# Create submission path and initialize submission class
submission_path = "submission"
create_submission = CreateSubmission(combined_model, test_dataloader, 0.25, submission_path)

# Create and save the predictions
create_submission.create_and_save_predictions()

# Load the created predictions
mask_path = os.path.join(submission_path, "prediction", "predictions")
mask_filenames = os.listdir(mask_path)
mask_filenames = [os.path.join(mask_path, f) for f in mask_filenames]

# Define submission name and save csv file
submission_name = os.path.join(submission_path, "final_submission.csv")
create_submission.masks_to_submission(submission_name, *mask_filenames)