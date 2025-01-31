import os
from typing import Tuple, Union, Optional
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImagePreprocessor:
    """Handles loading and preprocessing of images for adversarial attacks."""
    
    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """Initialize the image preprocessor.
        
        Args:
            image_size: Target size (height, width) for the image
            mean: Normalization mean for each channel (RGB)
            std: Normalization standard deviation for each channel (RGB)
        """
        self.size = size
        self.mean = mean
        self.std = std
        
    def image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return transform(image)
        
    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor back to a PIL Image.
        
        Args:
            tensor: Input tensor [1, C, H, W] or [C, H, W]
            denormalize: Whether to denormalize the tensor
            
        Returns:
            PIL Image
        """
        transform = transforms.Compose([
            transforms.Normalize(
                mean=[-m/s for m, s in zip(self.mean, self.std)],
                std=[1/s for s in self.std]
            )
        ])
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
            
        tensor = transform(tensor)
            
        return transforms.ToPILImage()(tensor)

    def load_image(
            self, 
            image_path: Union[str, Path], 
            return_pil: bool = False
        ) -> Union[torch.Tensor, Image.Image]:
            """Load and preprocess an image from a file path.
            
            Args:
                image_path: Path to the image file
                return_pil: If True, returns the PIL image before transformation
                
            Returns:
                Preprocessed image tensor [1, C, H, W] or PIL Image if return_pil=True
                
            Raises:
                FileNotFoundError: If the image file doesn't exist
                ValueError: If the image can't be opened or processed
            """
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
                
            try:
                image = Image.open(image_path).convert('RGB')
                
                if return_pil:
                    return image
                    
                tensor = self.image_to_tensor(image).unsqueeze(0)
                return tensor
                
            except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}")

    def save_image(
        self,
        tensor: torch.Tensor,
        save_path: Union[str, Path],
    ) -> None:
        """Save a tensor as an image file.
        
        Args:
            tensor: Image tensor to save
            save_path: Path where to save the image
            denormalize: Whether to denormalize before saving
        """
        image = self.tensor_to_image(tensor)
        image.save(save_path)