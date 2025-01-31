#%%
import sys
import os
import torch
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import ModelLoader, ImageNetLabels
from src.data_preproc import ImagePreprocessor
from src.adversarial import FGSMAttack
from pathlib import Path

#%%

preprocessor = ImagePreprocessor()
model_loader = ModelLoader(model_name='resnet18')
labels_path = Path('../data/imagenet_labels.json')
labels = ImageNetLabels(labels_path)

#%%

image_path = Path('../data/chimpanzee.jpg')
image = preprocessor.load_image(image_path, return_pil=True)
plt.imshow(image)
plt.show()
image = preprocessor.load_image(image_path, return_pil=False)
#%% Make a prediction with the model

predictions = model_loader.predict(image)
print(predictions)
predicted_label = labels.get_label(str(predictions[0][0][0]))
print(predicted_label)

#%%
# Iguana is class 39
target = torch.tensor([39])
attack = FGSMAttack(model_loader.model, epsilon=0.01)
adversarial_tensor = attack.generate(image, target=target)
adversarial_predictions = model_loader.predict(adversarial_tensor)
adversarial_label = labels.get_label(str(adversarial_predictions[0][0][0]))
print(adversarial_label)

# %% Plot adversarial image
adversarial_image = preprocessor.tensor_to_image(adversarial_tensor)
plt.imshow(adversarial_image)
plt.title(f"Adversarial image: {adversarial_label}")
plt.show()

# %%
