from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader


class SegFormer:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_labels=1, image_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Initialize the feature extractor
        self.feature_extractor = SegformerFeatureExtractor(
            do_normalize=True, do_resize=True, size=image_size
        )
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for the model using the feature extractor.
        :param image: PIL Image or numpy array
        :return: Preprocessed tensor
        """
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        return encoding["pixel_values"].to(self.device)
    
    def preprocess_mask(self, mask, image_size):
        """
        Preprocess a ground truth mask.
        :param mask: PIL Image or numpy array
        :param image_size: Tuple of (height, width) to resize the mask
        :return: Preprocessed tensor
        """
        # Resize mask to match 512x512 input size of model
        mask = mask.resize((512, 512))  
        # Normalize the mask
        mask = torch.tensor((np.array(mask) / 255.0), dtype=torch.float)
        return mask.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def train(self, dataloader, criterion, epochs=10, learning_rate=1e-4):
        """
        Fine-tune the SegFormer model on a dataset.
        :param dataloader: DataLoader object providing image-mask pairs
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for the optimizer
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = criterion

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
                
                # Compute loss
                loss = criterion(logits, masks)
                epoch_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    def predict(self, image):
        """
        Perform prediction on a single image.
        :param image: PIL Image or numpy array
        :return: Predicted mask (numpy array)
        """
        self.model.eval()
        with torch.no_grad():
            pixel_values = self.preprocess_image(image)
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits  # Shape: [batch_size, num_labels, height, width]
            predicted_mask = torch.sigmoid(logits).squeeze().cpu().numpy()  # Sigmoid for binary
            return (predicted_mask > 0.5).astype(np.uint8)  # Threshold to binary mask