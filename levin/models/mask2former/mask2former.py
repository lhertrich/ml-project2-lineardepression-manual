from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
import torch.optim as optim
import torch.nn.functional as F


class Mask2Former:
    def __init__(self, model_name="facebook/mask2former-swin-small-ade-semantic", original_size=(400,400)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep track of losses
        self.losses = {}
        self.validation_losses = {}
        self.id2label = {0: "background", 1: "road"}
        # Initialize the model
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic",
                                                          id2label=self.id2label,
                                                          ignore_mismatched_sizes=True).to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.original_size=original_size
    

    def validate(self, validationloader):
        """
        Compute validation loss over the validation set.
        """
        self.model.eval()
        validation_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in validationloader:
                pixel_values = batch["pixel_values"].to(self.device)  # Input images
                pixel_mask = batch["pixel_mask"].to(self.device)  # Mask for valid pixels
                mask_labels = batch["mask_labels"].to(self.device)  # Ground truth masks
                class_labels = batch["class_labels"].to(self.device)  # Ground truth class labels
            
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels
                )
                loss = outputs.loss
                validation_loss += loss.item()
                total_batches += 1
        # Return average validation loss
        return validation_loss / total_batches


    def train(self, dataloader, validationloader, save_path, epochs=10, learning_rate=1e-4):
        """
        Fine-tune the SegFormer model on a dataset.
        :param dataloader: DataLoader object providing image-mask pairs
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for the optimizer
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()

        model_paths = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            total_batches = 0
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)  # Input images
                pixel_mask = batch["pixel_mask"].to(self.device)  # Mask for valid pixels
                mask_labels = batch["mask_labels"].to(self.device)  # Ground truth masks
                class_labels = batch["class_labels"].to(self.device)  # Ground truth class labels
            
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels
                )

                loss = outputs.loss
                epoch_loss += loss.item()
                total_batches += 1

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
            avg_loss = epoch_loss / total_batches
            self.losses[epoch + 1] = avg_loss
            avg_validation_loss = self.validate(validationloader)
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
            prediction = self.model(pixel_values=pixel_values)
            target_sizes = [self.original_size] * pixel_values.size(0)
            prediction_mask = self.image_processor.post_process_semantic_segmentation(
                prediction, target_sizes=target_sizes
            )

        return prediction_mask