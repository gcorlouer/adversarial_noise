import pytest
import os
import sys
from PIL import Image
from pathlib import Path
from src.data_preproc import ImagePreprocessor
from src.model_utils import ModelLoader, ImageNetLabels

class TestImagePreprocessor:

    def test_image_to_tensor(self):
        preprocessor = ImagePreprocessor()
        image_path = Path('../data/frogs.jpg')
        image = preprocessor.load_image(image_path, return_pil=False)
        assert image.shape == (1, 3, 224, 224), f"Image shape is {image.shape}, should be {(3, 224, 224)}"
