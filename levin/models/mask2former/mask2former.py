from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
import torch.optim as optim
import torch.nn.functional as F


class Mask2Former:
    def __init__(self, model_name="facebook/mask2former-swin-small-ade-semantic", num_labels=2, original_size=(400,400)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep track of losses
        self.losses = {}
        self.validation_losses = {}

        # Initialize the model
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        self.original_size=original_size
    

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
                logits = outputs.masks_queries_logits
                logits = logits.sum(dim=1).unsqueeze(1)
                resized_logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

                loss = criterion(resized_logits, masks)
                validation_loss += loss.item() * len(images)
                total_val_samples += len(images)
        # Return average validation loss
        return validation_loss / total_val_samples


    def train(self, dataloader, validationloader, criterion, save_path, epochs=10, learning_rate=1e-4):
        """
        Fine-tune the SegFormer model on a dataset.
        :param dataloader: DataLoader object providing image-mask pairs
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for the optimizer
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = criterion
        self.model.train()

        model_paths = []
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
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)  # Aggregate queries
                masks = masks.unsqueeze(1)

                # Compute loss
                loss = criterion(logits, masks.float())
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

            epoch_save_path = save_path.replace(".pt", f"_epoch{epoch + 1}.pt")
            torch.save(self.model.state_dict(), epoch_save_path)
            model_paths.append(epoch_save_path)
            print(f"Model saved to {epoch_save_path}")

            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")

        print(f"Training finished")
        
    
    def predict(self, pixel_values):
        """
        Perform prediction on a single image.
        :param image: pixel_values
        :return: Predicted mask torch tensor
        """
        self.model.eval()
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device)
            # Forward pass
            outputs = self.model(pixel_values=pixel_values)

            # Apply threshold to get binary masks
            pred_semantic_map = self.image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[self.original_size]
            )[0]

        return pred_semantic_map