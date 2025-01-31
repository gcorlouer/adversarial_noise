import pytest
import os
import sys
import torch

from PIL import Image
from pathlib import Path
from src.data_preproc import ImagePreprocessor
from src.model_utils import ModelLoader, ImageNetLabels
from src.adversarial import FGSMAttack

class TestModelPredictions:

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


class TestAdversarialAttack:

    def test_fgsm_attack(self):
        preprocessor = ImagePreprocessor()
        model_loader = ModelLoader(model_name='resnet18')
        image_path = Path('../data/chimpanzee.jpg')
        labels_path = Path('../data/imagenet_labels.json')
        labels = ImageNetLabels(labels_path)
        model = model_loader.model
        attack = FGSMAttack(model, epsilon=0.001)
        image = preprocessor.load_image(image_path, return_pil=False)
        # 39 is the index of the 'iguana' class
        target = torch.tensor([39])
        target_label = labels.get_label(str(target.item()))
        adversarial_image = attack.generate(image, target)
        assert adversarial_image.shape == (1, 3, 224, 224), f"Adversarial image shape is {adversarial_image.shape}, should be {(3, 224, 224)}"
        adversarial_predictions = model_loader.predict(adversarial_image)
        label = labels.get_label(str(adversarial_predictions[0][0][0]))
        assert label == target_label, f"Adversarial class is {label}, should be {target_label}"
