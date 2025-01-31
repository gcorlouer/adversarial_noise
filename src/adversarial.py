import torch
import torch.nn.functional as F
from typing import Optional
from src.data_preproc import ImagePreprocessor

class AdversarialAttack(ImagePreprocessor):
    def __init__(
        self, 
        model, 
        epsilon=0.3,
        size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        super().__init__(size=size, mean=mean, std=std)
        self.model = model
        self.epsilon = epsilon 
        self.device = next(model.parameters()).device


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method adversarial attack."""
    
    def generate(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate adversarial example using FGSM.
        
        Args:
            x: Input tensor [1, C, H, W]
            target: Target class index as tensor, e.g., torch.tensor([386])
                   None for untargeted attack
            
        Returns:
            Adversarial example tensor
        """
        # Enable gradient tracking
        x.requires_grad = True
        
        # Forward pass
        output = self.model(x)
        
        # Calculate loss
        if target is None:
            # Untargeted attack: maximize loss for correct class
            loss = -F.cross_entropy(output, output.argmax(dim=1))
        else:
            # Targeted attack: minimize loss for target class
            target = target.to(self.device)
            loss = F.cross_entropy(output, target)
            
        # Backward pass
        loss.backward()
        
        # Create perturbation using gradient sign
        perturbation = self.epsilon * x.grad.sign()
        
        # Create adversarial example
        x_adv = x + perturbation
        
        # Ensure valid image range
        x_adv = torch.clamp(x_adv, -self.mean[0]/self.std[0], (1-self.mean[0])/self.std[0])
        
        return x_adv.detach()
