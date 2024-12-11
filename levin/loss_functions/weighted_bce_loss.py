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
    return self.loss(logits, targets)