import torch

class TverskyLoss(torch.nn.Module):
  def __init__(self, alpha=0.7, beta=0.3, epsilon=1e-6):
    """ Instantiates a tversky loss object
    
        Args:
            alpha = 0.7: float, the weight applied to false positives
            beta = 0.3: float, the weight applied to false negatives
            epsilon = 1e-6: float, small constant to ensure the function is defined in edge cases
    """
    super(TverskyLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon

  def forward(self, logits, targets):
    # Get probabilites
    probs = torch.sigmoid(logits)
    # Calculate true positives, false positives and false negatives 
    tp = (probs * targets).sum(dim=(2, 3))
    fp = ((1 - targets) * probs).sum(dim=(2, 3))
    fn = (targets * (1 - probs)).sum(dim=(2, 3))
    # Calculate tversky loss
    tversky = (tp + self.epsilon) / (tp + self.alpha * fp + self.beta * fn + self.epsilon)
    return 1 - tversky.mean()