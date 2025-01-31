#%%
import torch

from src.model_utils import ModelLoader, ImageNetLabels
from src.data_preproc import ImagePreprocessor
from src.adversarial import FGSMAttack
from pathlib import Path

#%%

preprocessor = ImagePreprocessor()
model_loader = ModelLoader(model_name='resnet18')
labels = ImageNetLabels(labels_path='data/imagenet_labels.json')

#%%

image_path = Path('data/chimpanzee.jpg')
image = preprocessor.load_image(image_path, return_pil=False)

#%%
target = torch.tensor([39])
attack = FGSMAttack(model_loader.model)
adversarial_image = attack.generate(image, target=target)

#%%

