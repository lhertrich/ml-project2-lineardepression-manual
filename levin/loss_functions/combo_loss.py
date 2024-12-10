import torch

class ComboLoss(torch.nn.Module):
  def __init__(self, weight=0.5, epsilon=1e-6):
    """ Instantiates a combo loss object
    
        Args:
            weight = 0.5: float, the weight applied to the bce loss
            epsilon = 1e-6: float, small constant to ensure the function is defined in edge cases
    """
    super(ComboLoss, self).__init__()
    self.weight = weight
    self.epsilon = epsilon
    self.bce = torch.nn.BCEWithLogitsLoss()

  def forward(self, logits, targets):
    # BCE Loss
    bce_loss = self.bce(logits, targets)

    # Dice Loss
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + self.epsilon
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.epsilon
    dice_loss = 1 - (num / den).mean()
    # Weighted combination of BCE and Dice loss
    return self.weight * bce_loss + (1 - self.weight) * dice_loss