import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim

class DeepLabV3Plus():
    def __init__(self, encoder_name="resnet50", encoder_weight="imagenet", in_channels=3, num_labels=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep track of losses
        self.losses = {}
        self.validation_losses = {}

        # Initialize the model
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,        # Encoder architecture
            encoder_weights=encoder_weight,  # Pretrained weights for encoder
            in_channels=in_channels,         # Input channels (3 for RGB)
            classes=num_labels,              # Number of output classes
        ).to(self.device)
    

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
                outputs = self.model(images)
                resized_outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)

                masks = masks.contiguous().float()

                # Compute loss
                loss = criterion(resized_outputs, masks)
                validation_loss += loss.item() * len(images)
                total_val_samples += len(images)

        # Return average validation loss
        return validation_loss / total_val_samples


    def train(self, dataloader, validationloader, criterion, save_path, epochs=10, learning_rate=1e-4, save=True):
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
            total_samples = 0
            for batch in dataloader:
                images, masks = batch

                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                resized_outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)

                masks = masks.contiguous().float()

                # Compute loss
                loss = criterion(resized_outputs, masks)
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

            if save:
                epoch_save_path = save_path.replace(".pt", f"_epoch{epoch + 1}.pt")
                torch.save(self.model.state_dict(), epoch_save_path)
                print(f"Model saved to {epoch_save_path}")

            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")

        print(f"Training finished")
        
    
    def predict(self, images, threshold=0.5):
        """
        Perform prediction on a single image.
        :param image: pixel_values
        :return: Predicted mask torch tensor
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            images = images.to(self.device)
            outputs = self.model(images)

            # Apply sigmoid activation to get probabilities
            probabilities = torch.sigmoid(outputs)

            # Apply threshold to get binary masks
            binary_masks = (probabilities > threshold).to(torch.uint8)

        return binary_masks