import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import json
from .helpers import patch_to_label
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class TrainAndEvaluate():
    def __init__(self, model, train_loader, validation_loader, test_loader, epochs, save_dir, save=False):
        """Initializes an object to train and evaluate a given model"""
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.save_dir = save_dir
        self.model_save_path = os.path.join(self.save_dir, "models", f"trained_model.pt")
        self.save = save
        self.evaluation_metrics = {}

    
    def evaluate_model(self, patch_size=16, threshold=0.25):
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
            for batch in self.test_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                predictions = self.model(pixel_values=pixel_values)
                target_sizes = [(mask.shape[0], mask.shape[1]) for mask in batch["original_mask"]]
                predicted_masks = self.model.image_processor.post_process_semantic_segmentation(predictions, target_sizes=target_sizes)
                original_masks = batch["original_mask"]

                for pred, mask in zip(predicted_masks, original_masks):
                    # Remove the singleton dimension
                    pred = pred.cpu().numpy()
                    mask = mask.cpu().numpy()

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
    

    def plot_losses(self, train_losses, val_losses):
        """
        Plot and save training and validation losses

        Args:
            train_losses: dictionary, epoch-wise training losses
            val_losses: dictionary, epoch-wise validation losses
        """
        # Extract epochs and losses
        epochs = list(train_losses.keys())
        train_loss_values = list(train_losses.values())
        val_loss_values = list(val_losses.values())

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss_values, label="Training Loss", marker='o')
        plt.plot(epochs, val_loss_values, label="Validation Loss", marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(self.model_dir, "loss_over_epochs.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Loss plot saved to {plot_path}")
    

    def save_predictions(self, num_images=5):
        """ Saves a given number of predictions for test images

        Args:
            num_images = 5: int, the number of test images for which predictions should be saved
        """
        predictions_dir = os.path.join(self.model_dir, "sample_predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        saved_count = 0
        with torch.no_grad():
            for images, masks in self.test_loader:
                if saved_count >= num_images:
                    break

                # Move to device
                images = images.to(self.device)
                masks = masks.cpu().numpy()

                # Get predictions
                preds = self.model.predict(images).cpu().numpy()

                for idx in range(len(images)):
                    if saved_count >= num_images:
                        break
                    image = images[idx].cpu().squeeze()
                    maski = masks[idx].squeeze().squeeze()
                    predi = preds[idx].squeeze().squeeze()
                    # Save input image, ground truth mask, and predicted mask
                    img = TF.to_pil_image(image)
                    mask = TF.to_pil_image((maski * 255).astype(np.uint8))
                    pred = TF.to_pil_image((predi * 255).astype(np.uint8))

                    img.save(os.path.join(predictions_dir, f"test_image_{saved_count}.png"))
                    mask.save(os.path.join(predictions_dir, f"test_mask_{saved_count}.png"))
                    pred.save(os.path.join(predictions_dir, f"pred_mask_{saved_count}.png"))
                    saved_count += 1

                
    def run(self):
        """ Trains the model, evaluates it and saves training and evaluation losses as well as sample outputs

            Returns:
                best_f1: float, the best f1 score of the model on the patched predictions after self.epochs epochs
                best_threshold: float, the threshold used to achieve the best f1 score
                best_epoch: int, the epoch with the smalles evaluation loss
        """
        # Train model
        self.model.train(self.train_loader, self.validation_loader, self.model_save_path, epochs=self.epochs)
        print("Training complete")

        # Find best epoch based on evaluation loss
        best_epoch = min(self.model.validation_losses, key=self.model.validation_losses.get)
        print(f"Best epoch: {best_epoch}")

        # Load the best model
        best_model_path = self.model_save_path.replace(".pt", f"_epoch{best_epoch}.pt")
        self.model.model.load_state_dict(torch.load(best_model_path))
        print(f"Best model loaded from {best_model_path}")

        # Delete other models
        for epoch in range(1, self.epochs + 1):
            model_path = self.model_save_path.replace(".pt", f"_epoch{epoch}.pt")
            if model_path != best_model_path and os.path.exists(model_path):
                os.remove(model_path)
                print(f"Deleted model: {model_path}")

        # Evaluate the model on the test set for different thresholds
        thresholds = [0.25, 0.5]
        best_f1 = 0
        best_threshold = 0
        for threshold in thresholds:
            f1, acc, prec, rec = self.evaluate_model(patch_size=16, threshold=threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            self.evaluation_metrics[f"{threshold:.2f}"] = {"f1_score": f1, "accuracy": acc, "precision": prec, "recall": rec}
            print(f"Threshold {threshold}, F1: {f1:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

        # Save evaluation metrics to JSON
        metrics_path = os.path.join(self.model_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.evaluation_metrics, f, indent=4)

        # Plot training and evaluation losses
        self.plot_losses(self.model.losses, self.model.validation_losses)

        # Save predictions for 5 test images
        self.save_predictions()

        print(f"Everything complete. Results saved to {self.model_dir}.")
        return best_f1, best_threshold, best_epoch