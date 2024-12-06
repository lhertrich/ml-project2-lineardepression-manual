from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class SegFormer:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_labels=1, image_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Reinitialize the decode head classifier for binary segmentation
        self.model.decode_head.classifier = torch.nn.Conv2d(
            in_channels=256,  # Feature dimension
            out_channels=num_labels,  # Number of output labels
            kernel_size=1
        ).to(self.device)
        self.model.decode_head.bias = torch.nn.Parameter(torch.zeros(num_labels)).to(self.device)

        # Initialize the feature extractor
        self.feature_extractor = SegformerImageProcessor(
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
    
    def preprocess_mask(self, mask):
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

            # Inspect intermediate shapes (optional, based on model internals)
            print(f"Decoder head input shape: {self.model.decode_head.hidden_states.shape}")
            print(f"Decoder head output shape: {self.model.decode_head.logits.shape}")
    
    def train(self, dataloader, criterion, epochs=10, learning_rate=1e-4, save_path="segformer.pt"):
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
                resized_logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)
                resized_logits = resized_logits.reshape(masks.shape)

                masks = masks.contiguous().float()

                # Compute loss
                loss = criterion(resized_logits, masks)
                epoch_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def predict(self, image):
        """
        Perform prediction on a single image.
        :param image: PIL Image or numpy array
        :return: Predicted mask (numpy array)
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=image)
            logits = outputs.logits  # Shape: [batch_size, num_labels, height, width]
            probabilities = torch.sigmoid(logits).cpu().numpy()  # Sigmoid for binary
            return (probabilities > 0.5).astype(np.uint8)  # Threshold to binary mask
