from levin.SegFormer.segformer_b0 import SegFormer
from levin.dataset import RoadSegmentationDataset
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
import torchvision.transforms as T
import os
import torch
#from sklearn.model_selection import train_test_split


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Construct the absolute path to the data folder
image_dir = os.path.join(base_dir, "data", "training", "augmented", "images")
mask_dir = os.path.join(base_dir, "data", "training", "augmented", "masks")

print(f"Image directory: {image_dir}")
print(f"Mask directory: {mask_dir}")

# Check if the paths exist
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")
if not os.path.exists(mask_dir):
    raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

# Define transforms for images and masks
feature_extractor = SegformerImageProcessor(
    do_normalize=True,
    do_resize=True,
    size=512
)

mask_transform = T.Compose([
    T.Resize((512, 512)),  # Resize masks to 512x512
    T.ToTensor()           # Convert to PyTorch tensor (binary mask stays as float)
])

image_filenames = sorted(os.listdir(image_dir))
mask_filenames = sorted(os.listdir(mask_dir))

# Split filenames into train and test sets
#train_images, test_images, train_masks, test_masks = train_test_split(
#    image_filenames, mask_filenames, test_size=0.2, random_state=42
#)

# Define datasets for train and test sets
train_dataset = RoadSegmentationDataset(
    image_dir=image_dir, 
    mask_dir=mask_dir, 
    image_filenames=image_filenames,
    mask_filenames=mask_filenames,
    feature_extractor=feature_extractor, 
    mask_transform=mask_transform
)

# test_dataset = RoadSegmentationDataset(
#     image_dir=image_dir, 
#     mask_dir=mask_dir, 
#     transform=image_transform, 
#     target_transform=mask_transform
# )

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

criterion = torch.nn.BCEWithLogitsLoss()
save_path = "/trained_models/segformer_default_BCE.pt"
model = SegFormer()
model.train(train_loader, criterion, epochs=10, save_path=save_path)