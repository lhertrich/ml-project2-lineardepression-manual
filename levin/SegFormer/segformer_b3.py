from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class SegFormer:
    def __init__(self, model_name="nvidia/segformer-b3-finetuned-ade-512-512", num_labels=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Keep track of losses
        self.losses = {}

        # Initialize the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Reinitialize the decode head classifier for binary segmentation
        self.model.decode_head.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=768,  # Feature dimension from the encoder
                out_channels=num_labels,  # Number of output labels
                kernel_size=1,  # 1x1 convolution to map features to logits
            ),
            torch.nn.Upsample(
                size=(512, 512),  # Upsample to the target resolution
                mode='bilinear',
                align_corners=False
            )
        ).to(self.device)
    

    def debug_model_output(self, pixel_values):
        """
        Debug model to inspect intermediate output shapes.
        :param pixel_values: Tensor of shape [batch_size, num_channels, height, width]
        """
        with torch.no_grad():
            self.model.eval()

            # Forward pass through the model
            outputs = self.model(pixel_values=pixel_values)

            # Logits are the final output of the model
            logits = outputs.logits
            print(f"Logits shape: {logits.shape}")  # Shape should be [batch_size, num_labels, 512, 512]
    
    def train(self, dataloader, criterion, epochs=10, learning_rate=1e-4, save_path="segformer.pt"):
        """
        Fine-tune the SegFormer model on a dataset.
        :param dataloader: DataLoader object providing image-mask pairs
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for the optimizer
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = criterion
        epoch_losses = {}
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
                resized_logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

                masks = masks.contiguous().float()

                # Compute loss
                loss = criterion(resized_logits, masks)
                epoch_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss /= sum(len(batch[0]) for batch in dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            epoch_losses[epoch+1] = epoch_loss

        torch.save(self.model.state_dict(), save_path)
        self.losses = epoch_losses
        print(f"Model saved to {save_path}")
    
    def predict(self, pixel_values, threshold=0.5):
        """
        Perform prediction on a single image.
        :param image: pixel_values
        :return: Predicted mask torch tensor
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            outputs = self.model(pixel_values=pixel_values)  # Assumes pixel_values is preprocessed
            logits = outputs.logits  # Shape: [batch_size, num_labels, height, width]

            # Apply sigmoid activation to get probabilities
            probabilities = torch.sigmoid(logits)

            # Apply threshold to get binary masks
            binary_masks = (probabilities > threshold).to(torch.uint8)  # Convert to uint8 for binary masks

        return binary_masks  # Return as Torch tensor