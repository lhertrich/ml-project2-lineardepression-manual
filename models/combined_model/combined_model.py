import torch


class CombinedModels:
    def __init__(self, model_deep_lab, model_segformer):
        """ Initialized a combined model object, that combines a DeepLabV3Plus and a Segformer model

        Args:
         model_1: DeepLabV3Plus model, existing model that should be combined
         model_2: SegFormerModel, existing model that should be combined 
        """
        self.model_deep_lab = model_deep_lab
        self.model_segformer = model_segformer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, input, threshold=0.5):
        """ Performs prediction on a batch of images

        Args:
            pixel_values: torch.Tensor, input images of shape (batch_size, channels, height, width)
            threshold = 0.5: float, threshold value for converting probabilities to binary masks

        Returns:
            torch.Tensor, predicted binary masks of shape (batch_size, height, width), with values in {0, 1}.
        """
        self.model_deep_lab.model.eval()
        self.model_segformer.model.eval()
        with torch.no_grad():
            deeplab_image, pixel_values = input
            deeplab_image = deeplab_image.to(self.device)
            pixel_values = pixel_values.to(self.device)

            # Forward pass for DeepLabV3+
            outputs_deeplab = self.model_deep_lab.model(deeplab_image)  
            # Apply sigmoid activation to get probabilities
            probabilities_deeplab = torch.sigmoid(outputs_deeplab)

            # Forward pass for Segformer
            outputs_segformer = self.model_segformer.model(pixel_values=pixel_values)
            logits_segformer = outputs_segformer.logits
            # Apply sigmoid activation to get probabilities
            probabilities_segformer = torch.sigmoid(logits_segformer)

            # Combine predictions
            combined_probabilities = 0.3 * probabilities_deeplab + 0.7 * probabilities_segformer
            binary_masks = (combined_probabilities > threshold).to(torch.uint8)
        return binary_masks