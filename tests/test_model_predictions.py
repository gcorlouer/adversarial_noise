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
    
    def test_model_prediction(self):
        preprocessor = ImagePreprocessor()
        model = ModelLoader(model_name='resnet18')
        image_path = Path('../data/chimpanzee.jpg')
        labels_path = Path('../data/imagenet_labels.json')
        labels = ImageNetLabels(labels_path)
        image = preprocessor.load_image(image_path, return_pil=False)
        predictions = model.predict(image)
        label = labels.get_label(str(predictions[0][0][0]))
        proba = predictions[1][0][0]
        assert label == 'chimpanzee, chimp, Pan troglodytes', f"Predicted class is {label}, should be chimpanzee, chimp, Pan troglodytes"
        assert proba < 1.0, f"Predicted probability is {proba}, should be less than 1.0"
        assert proba >= 0.9, f"Predicted probability is {proba}, should be greater than 0.9"



