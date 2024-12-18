import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim

class DeepLabV3Plus():
    def __init__(self, encoder_name="resnet50", encoder_weight="imagenet", in_channels=3, num_labels=1):
        """Initializes a DeepLabV3+ model object
        
        Args:
            encoder_name = 'resnet50': string, the name of the encoder for the model
            encoder_weight = 'imagenet': string, the name of the dataset chosen for the pretrained weights
            in_channels = 3: int, the number of input channels for the model (e.g. 3 for RGB images)
            num_labels = 1: int, the number of classes to predict
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep track of losses
        self.losses = {}
        self.validation_losses = {}

        # Initialize the model
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,   
            encoder_weights=encoder_weight,
            in_channels=in_channels,
            classes=num_labels,
        ).to(self.device)
    

    def validate(self, validationloader, criterion):
        """ Computes validation loss over the validation set

        Args:
            validationloader: DataLoader, the data loader with validation images and masks
            criterion: Object, the loss function

        Returns:
            float, average validation loss
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
        """ Trains the DeepLabV3+ model on a given dataset
    
        Args:
            dataloader: DataLoader, dataLoader object providing training image-mask pairs
            validationloader: DataLoader dataLoader object providing validation image-mask pairs
            criterion: Object, the loss function to optimize
            save_path: str, file path to save the model weights
            epochs = 10: int, number of training epochs
            learning_rate = 1e-4: float, learning rate for the optimizer
            save = True: boolean, whether to save the model weights after each epoch
    
        Prints:
            Training and validation loss for each epoch

        Saves:
            The model's state_dict to `save_path` with epoch-specific filenames if `save` is True
        """
        print("Starting training")
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
                # Resize logits to match groundtruth to compute loss
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
                
            # Calculate and store losses
            avg_loss = epoch_loss / total_samples
            self.losses[epoch + 1] = avg_loss
            avg_validation_loss = self.validate(validationloader, criterion)
            self.validation_losses[epoch + 1] = avg_validation_loss

            # Save model if specified
            if save:
                epoch_save_path = save_path.replace(".pt", f"_epoch{epoch + 1}.pt")
                torch.save(self.model.state_dict(), epoch_save_path)
                print(f"Model saved to {epoch_save_path}")

            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")

        print(f"Training finished")
        
    
    def predict(self, images, threshold=0.5):
        """Performs prediction on a batch of images

        Args:
            images: torch.Tensor, input images of shape (batch_size, channels, height, width)
            threshold = 0.5: float, threshold value for converting probabilities to binary masks

        Returns:
            torch.Tensor, predicted binary masks of shape (batch_size, height, width), with values in {0, 1}.
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