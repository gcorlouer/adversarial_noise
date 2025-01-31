# Adversarial Image Generator
A library for generating adversarial examples to test the robustness of image classification models.

Installation

`pip install -r requirements.txt`

See examples/demo.py script for a an example using the codebase to produce adversarial attack.

The current implementation fails to fool the model. This might be because the adversarial attack is too naive. I varied the epsilon value but this did not change the results.

# Next steps:

* Make sure that the problem does not come from clamping the image in the adversarial attack generation 
* Try more sophisticated adversarial attackes like project gradient descent
* Spend more time playing with hyperparameters (ex: values of epsilon, number of iterations in PGD)
* Add utility functions to get the correct class label from class name provided by the user using imagenet_label

