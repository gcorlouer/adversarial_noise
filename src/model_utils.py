import torch
import torchvision.models as models
import logging
import json

from typing import Tuple, List, Dict, Optional
from pathlib import Path
from data_preproc import ImagePreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and inference for pretrained image classification models."""

    SUPPORTED_MODELS = {
        "resnet18": models.resnet18,
    }

    def __init__(self, model_name: str = "resnet18", device: Optional[str] = None):
        """Initialize the model loader.

        Args:
            model_name: Name of the pretrained model to use
            device: Device to run the model on ('cuda' or 'cpu'). If None, automatically detects.

        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self._setup_model()

    def _setup_model(self) -> None:
        """Set up the pretrained model."""
        try:
            # Load pretrained model
            model_fn = self.SUPPORTED_MODELS[self.model_name]
            self.model = model_fn(weights=models.ResNet18_Weights.DEFAULT)

            # Move to appropriate device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Successfully loaded {self.model_name} on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(
        self, input_tensor: torch.Tensor, top_k: int = 1
    ) -> Tuple[List[int], List[float]]:
        """Make a prediction on a preprocessed input tensor.

        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            top_k: Number of top predictions to return

        Returns:
            Tuple containing:
                - List of top-k predicted class indices
                - List of corresponding probabilities

        Raises:
            RuntimeError: If model is not initialized
            ValueError: If input tensor has incorrect shape
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Validate input
        if len(input_tensor.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor [1,C,H,W], got shape {input_tensor.shape}"
            )

        try:
            # Move input to correct device
            input_tensor = input_tensor.to(self.device)

            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=-1)

            # Get top k predictions
            top_probs, top_classes = torch.topk(probabilities, top_k)

            return (top_classes.tolist(), top_probs.tolist())

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def get_model(self) -> torch.nn.Module:
        """Return the underlying PyTorch model."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model


class ImageNetLabels:
    """Handles ImageNet class label mapping."""

    def __init__(self, labels_path: Optional[str] = None):
        """Initialize with path to ImageNet labels JSON file.

        If no path provided, uses a minimal set of labels.
        """
        self.labels = self._load_labels(labels_path)

    def _load_labels(self, labels_path: Optional[str]) -> Dict[int, str]:
        if Path(labels_path).exists():
            with open(labels_path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Labels file {labels_path} does not exist")

    def get_label(self, idx: int) -> str:
        """Get human-readable label for ImageNet class index."""
        return self.labels.get(idx, f"Unknown class {idx}")
