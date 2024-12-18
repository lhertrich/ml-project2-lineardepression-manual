import os
import matplotlib.image as mpimg
import re
import torch
import numpy as np
import torch.nn.functional as F
import shutil
from utils.helpers import patch_to_label
from PIL import Image


class CreateSubmission():
    def __init__(self, model, test_dataloader, threshold, save_path):
        """ Initializes a CreateSubmission object

        Args:
            model: Object, the trained model that should be used for submission
            test_dataloader: Dataloader, the dataloader with the test images for submission
            threshold: int, the threshold which should be used to predict the patches
            save_path: string, the path where the submission files should be saved
        """
        self.model = model
        self.test_dataloader = test_dataloader
        self.threshold = threshold
        self.save_path = save_path
    

    def mask_to_submission_strings(self, image_filename):
        """Reads a single image and outputs the strings that should go into the submission file
        
        Args:
            image_filename: string, the filename of the image for which the strings are created

        Yields:
            str: A string in the format "{:03d}_{}_{}," where:
                - The first part is the zero-padded image number extracted from the filename
                - The second and third parts are the x and y coordinates of the patch (column and row indices)
                - The last part is the predicted label for the patch, based on the function `patch_to_label`
        """
        img_number = int(re.search(r"prediction_(\d+)", image_filename).group(1))
        print(f"Filename: {image_filename}, number: {img_number}")
        im = mpimg.imread(image_filename)
        patch_size = 16
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch, self.threshold)
                yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

            
    def masks_to_submission(self, submission_filename, *image_filenames):
        """Converts images into a submission file
        
        Args:
            submission_filename: string, the name of the file where to write the submission
            *image_filenames: iterable, the image_filenames of the masks which should be written into the submission file
        """
        print("Submission_filename: ", submission_filename)
        with open(submission_filename, 'w') as f:
            f.write('id,prediction\n')
            for fn in image_filenames[0:]:
                f.writelines('{}\n'.format(s) for s in self.mask_to_submission_strings(fn))


    def resize_predictions(self, predictions, target_size=(608, 608)):
        """ Resize a batch of predictions to the target size

        Args:
            predictions: Dataloader batch, batch of predicted masks (numpy array or tensor of shape [batch_size, 1, H, W])
            target_size = (608,608): tuple, target size for resizing

        Returns:
            numpy array: the resized predictions
        """
        # Convert numpy array to tensor if needed
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions)

        # Ensure the tensor is in float32 for resizing
        predictions = predictions.to(dtype=torch.float32)

        # Resize predictions using bilinear interpolation
        resized_predictions = F.interpolate(predictions, size=target_size, mode="nearest")

        return resized_predictions.cpu().numpy()
    

    def create_and_save_predictions(self):
        """Creates and saves predictions for images given to the class"""
        prediction_dir = os.path.join(self.save_path, "prediction", "predictions")
        prediction_image_dir = os.path.join(self.save_path, "prediction", "images")

        # Remove existing directories and their contents
        if os.path.exists(prediction_dir):
            shutil.rmtree(prediction_dir)
        if os.path.exists(prediction_image_dir):
            shutil.rmtree(prediction_image_dir)

        # Recreate directories
        os.makedirs(prediction_dir, exist_ok=True)
        os.makedirs(prediction_image_dir, exist_ok=True)

        # Counter to keep track of the prediction files
        prediction_count = 1

        with torch.no_grad():
            for batch in self.test_dataloader:

                # Get predictions
                predictions = self.model.predict(batch)

                # Resize predictions
                predictions = self.resize_predictions(predictions)

                # Iterate over each prediction in the batch
                for pred in predictions:
                    print("Saving prediction: ", prediction_count)
                    raw_pred_path = os.path.join(prediction_dir, f"prediction_{prediction_count}.png")
                    binary_pred = (pred.squeeze() * 255).astype(np.uint8)
                    binary_image = Image.fromarray(binary_pred)
                    binary_image.save(raw_pred_path)

                    # Save the black-and-white (scaled) prediction as a PNG
                    bw_pred = (pred.squeeze() * 255).astype(np.uint8)
                    bw_image = Image.fromarray(bw_pred)
                    bw_image_path = os.path.join(prediction_image_dir, f"prediction_image_{prediction_count}.png")
                    bw_image.save(bw_image_path)

                # Increment prediction counter
                prediction_count += 1

        print(f"Saved {prediction_count} predictions to {prediction_dir}")