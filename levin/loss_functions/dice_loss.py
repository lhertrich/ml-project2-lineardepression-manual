import torch

class DiceLoss(torch.nn.Module):
  def __init__(self, epsilon=1e-6):
    """ Instantiates a dice loss object
    
        Args:
            epsilon = 1e-6: float, small constant to ensure the function is defined in edge cases
    """
    super(DiceLoss, self).__init__()
    self.epsilon = epsilon

  def forward(self, logits, targets):
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    # Calculate numerator and denumerator
    num = 2 * (probs * targets).sum(dim=(2, 3)) + self.epsilon
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.epsilon
    dice_loss = 1 - (num / den)
    return dice_loss.mean()