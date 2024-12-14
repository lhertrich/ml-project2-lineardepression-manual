from transformers import SegformerForSemanticSegmentation
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from ...helpers import patch_to_label
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class SegFormer:
    def __init__(self, model_name="nvidia/segformer-b3-finetuned-ade-512-512", num_labels=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Keep track of losses
        self.losses = {}
        self.validation_losses = {}
        self.f1_scores = {}

        # Initialize the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Reinitialize the decode head classifier for binary segmentation
        self.model.decode_head.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=768, 
                out_channels=num_labels,
                kernel_size=1,
            ),
            torch.nn.Upsample(
                size=(512, 512),
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
    

    def validate(self, validationloader, criterion):
        """
        Compute validation loss over the validation set.
        """
        self.model.eval()
        validation_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch in validationloader:
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
                validation_loss += loss.item() * len(images)
                total_val_samples += len(images)

        # Return average validation loss
        return validation_loss / total_val_samples


    def evaluate_model(self, test_loader, patch_size=16, threshold=0.25):
        """ Evaluates the model on the given patchsize with the given threshold based on f1 score, accuracy, precision and recall

            Args:
                patch_size: int, the size of patches which are aggregated to a prediction
                threshold: float, the threshold for the decision how an aggregated patch is classified

            Returns:
                f1: float, the calculated f1 score
                accuracy: float, the calculated accuracy
                precision: float, the calculated precision
                recall: float, the calculated recall
        """
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks in test_loader:
                # Move masks to the correct device
                masks = masks.cpu().numpy()

                # Get predictions for the entire batch
                preds = self.predict(images)
                preds = preds.cpu().numpy()

                for pred, mask in zip(preds, masks):
                    # Remove the singleton dimension
                    pred = np.squeeze(pred)  
                    mask = np.squeeze(mask)

                    if pred.shape != mask.shape:
                        raise ValueError(f"Shape mismatch: pred {pred.shape}, mask {mask.shape}")

                    # Divide prediction and mask into patches
                    height, width = mask.shape
                    patch_preds = []
                    patch_targets = []

                    for y in range(0, height, patch_size):
                        for x in range(0, width, patch_size):
                            pred_patch = pred[y:min(y+patch_size, height), x:min(x+patch_size, width)]
                            mask_patch = mask[y:min(y+patch_size, height), x:min(x+patch_size, width)]

                            if pred_patch.size == 0 or mask_patch.size == 0:
                                print(f"Skipping empty patch at y={y}, x={x}")
                                continue

                            # Convert patch to binary label
                            pred_label = patch_to_label(pred_patch, threshold)
                            target_label = patch_to_label(mask_patch, threshold)

                            patch_preds.append(pred_label)
                            patch_targets.append(target_label)

                    all_preds.extend(patch_preds)
                    all_targets.extend(patch_targets)

        if not all_preds or not all_targets:
            raise ValueError("Predictions or targets are empty. Evaluation cannot proceed.")
        
        # Calculate metrics at the patch level
        f1 = f1_score(all_targets, all_preds, average="binary")
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average="binary")
        recall = recall_score(all_targets, all_preds, average="binary")
        return f1, accuracy, precision, recall


    def train(self, dataloader, validationloader, testloader, criterion, save_path, epochs=10, learning_rate=1e-4):
        """
        Fine-tune the SegFormer model on a dataset.
        :param dataloader: DataLoader object providing image-mask pairs
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for the optimizer
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = criterion
        self.model.train()
        best_f1 = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            total_samples = 0
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
                epoch_loss += loss.item() * len(images)
                total_samples += len(images)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            avg_loss = epoch_loss / total_samples
            self.losses[epoch + 1] = avg_loss
            avg_validation_loss = self.validate(validationloader, criterion)
            self.validation_losses[epoch + 1] = avg_validation_loss

            if epoch % 3 == 0:
                f1, _, _, _ = self.evaluate_model(testloader)
                self.f1_scores[epoch] = f1
                if f1 > best_f1:
                    best_f1 = f1
                    epoch_save_path = save_path.replace(".pt", f"_epoch{epoch + 1}.pt")
                    torch.save(self.model.state_dict(), epoch_save_path)
                    print(f"Model saved to {epoch_save_path}")
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}, F1: {f1:.4f}")

            else:
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")



        print(f"Training finished")
        
    
    def predict(self, pixel_values, threshold=0.5):
        """
        Perform prediction on a single image.
        :param image: pixel_values
        :return: Predicted mask torch tensor
        """
        self.model.eval()
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device)
            # Forward pass
            outputs = self.model(pixel_values=pixel_values)  # Assumes pixel_values is preprocessed
            logits = outputs.logits  # Shape: [batch_size, num_labels, height, width]

            # Apply sigmoid activation to get probabilities
            probabilities = torch.sigmoid(logits)

            # Apply threshold to get binary masks
            binary_masks = (probabilities > threshold).to(torch.uint8)  # Convert to uint8 for binary masks

        return binary_masks  # Return as Torch tensor