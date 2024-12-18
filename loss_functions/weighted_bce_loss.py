import torch

class WeightedBCEWithLogitsLoss(torch.nn.Module):
  def __init__(self, pos_weight=4.0):
    """ Instantiates a weighted BCE with logits loss object
    
    Args:
        pos_weight = 4.0: float, the weight for positive examples
    """
    super(WeightedBCEWithLogitsLoss, self).__init__()
    self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

  def forward(self, logits, targets):
    """Calculates the weighted BCE with logits loss
        
    Args:
        logits: torch.Tensor, predicted logits from the model, of shape (batch_size, num_classes, height, width)
        targets: torch.Tensor, ground truth labels, of shape (batch_size, num_classes, height, width)
        
    Returns:
        torch.Tensor: the calculated weighted BCE loss value
    """
    return self.loss(logits, targets)