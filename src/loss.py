import torch
import torch.nn as nn

class PreAveragedFocalLoss(nn.Module):
    """
    Pre-Averaged/Modified Focal Loss that applies the focal loss modification to the averaged cross-entropy loss.
    
    This specific variant of focal loss modifies the overall averaged loss rather than individual sample
    losses. Roles of alpha and gamma are generalized to the averaged loss context.

    Attributes:
        alpha (float): Serves as a scaling factor for the overall loss.
        gamma (float): Adjusts the emphasis on the overall difficulty of classification/confidence of the average predictions.
    
    Methods:
        forward(inputs, targets):
            Computes the pre-averaged/modified focal loss.
    """
    def __init__(
        self, 
        alpha: float = 1., 
        gamma: float = 2.
        ) -> None:
        """
        Initializes the PreAveragedFocalLoss class with the given alpha and gamma values.
        
        Parameters:
            alpha (float): Serves as a scaling factor for the overall loss.
            gamma (float): Adjusts the emphasis on the overall difficulty of classification.
        """
        super(PreAveragedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()  # Use reduction='mean' for this modification

    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
        ) -> torch.Tensor:
        """
        Computes the pre-averaged/modified focal loss.
        
        Parameters:
            inputs (torch.Tensor): The predictions from the model.
            targets (torch.Tensor): The ground truth labels.
        
        Returns:
            torch.Tensor: The computed pre-averaged/modified focal loss.
        """
        # Compute the average cross-entropy loss
        avg_ce_loss = self.ce(inputs, targets)
        # Use the averaged loss in a manner similar to individual log-probabilities
        p = torch.exp(-avg_ce_loss)
        # Apply the focal loss modification to the averaged loss
        loss = self.alpha * (1 - p) ** self.gamma * avg_ce_loss
        
        return loss
