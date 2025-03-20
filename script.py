
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

#Image processing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numbers

import math

import sys
sys.path.append('deep_dream/utilities')

from utility import *

     
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def deepdream_image_generator(PIL_image,model):

  tensor_image= pre_prcoess_PIL_to_np_to_tensor(PIL_image) # (B,C,H,W)
  # tensor_image.requires_grad=True # user created tensors has to be manually set this. But we do not need this here. If we start the graph from here backward will compute gradient till here.
  image_sizes= pyramid_ratio_generator(tensor_image,number_of_pyramids,pyramid_ratio)

  for i in image_sizes:

    tensor_image=transforms.Resize(i)(tensor_image)

    for i in range(iterations):

      h_shift, w_shift = np.random.randint(-shift, shift + 1, 2)

      with torch.no_grad():
        tensor_image = torch.roll(tensor_image, (h_shift,w_shift),(2,3))

      gradient_ascent(tensor_image,model)

      with torch.no_grad():
        tensor_image = torch.roll(tensor_image,(-h_shift, -w_shift),(2,3)) # this will output a leaf no requires_grad tensor breaking the computational graph connection to previous tensors

  return post_process_torch_to_numpy(tensor_image)

def gradient_ascent(tensor_image,model):
  tensor_image.requires_grad=True# the grads will fill till here. Retain_grad has to be called as method. requires_grad is called as func
  intermediate_activations= model.forward(tensor_image) # Therefore image must be set True for Two reasons 1) Model parameters are set requiers_grad =False so we need to do this to calculate gradients otherwise gradients will not be calulcated. 2)we are updating the image with gradients.
  if intermediate_activations[0].requires_grad==False:
    raise ValueError(" Intermediate activations do not have requires_grad true")
  loss=[]
  for i in intermediate_activations:
    loss.append(i.sum())
  loss=torch.stack(loss)
  loss= torch.sum(loss)
  loss.backward()
  grad= tensor_image.grad.data
  sigma = (iterations + 1) / iterations * 2.0 + 0.5
  smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma,DEVICE=device)(grad)
  # print(f'data/grad={abs(tensor_image.std()/ smooth_grad.std())}')
  # lr=  10**-10 * abs(tensor_image.std()/ smooth_grad.std())
  with torch.no_grad():
    tensor_image += lr * smooth_grad    # An inplace operation. This inplace operation is only allowed using  torch.no_grad().
  tensor_image.grad.zero_()


import torch.nn as nn

class PretrainedModel(nn.Module):
    def __init__(self, model, layer_names_or_indices, is_sequential=False):
        super(PretrainedModel, self).__init__()
        self.model = model
        self.layer_names_or_indices = layer_names_or_indices
        self.is_sequential = is_sequential
        self.feature_maps = [None] * len(layer_names_or_indices)

        # Register hooks for the appropriate layers based on the model type
        if self.is_sequential:
            # For Sequential models, treat layer_names_or_indices as indices
            for i, layer_idx in enumerate(self.layer_names_or_indices):
                layer = self.model.features[layer_idx]  # Access Sequential layers by index
                layer.register_forward_hook(self.create_hook(i))
        else:
            # For named layers, treat layer_names_or_indices as layer names
            for i, layer_name in enumerate(self.layer_names_or_indices):
                layer = getattr(self.model, layer_name)  # Access named layers
                layer.register_forward_hook(self.create_hook(i))

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def create_hook(self, index):
        def hook_fn(module, input, output):
            self.feature_maps[index] = output
        return hook_fn

    def forward(self, image):
        # Forward pass through the model
        _ = self.model(image)

        # Return the feature maps captured by the hooks as a tuple
        return tuple(self.feature_maps)

# Hyperparameters
shift=32 # How image is to be shifted during roll.
pyramid_ratio =1.8
number_of_pyramids = 4
iterations=10
lr= 0.005 # for VGG
# lr=0.5 # For facenet
# layer_name=('conv2d_4b', 'mixed_7a') # Facenet
layer_name = (22,) # VGG16
model = models.vgg16(weights= 'VGG16_Weights.IMAGENET1K_V1')
image=deepdream_image_generator(Image.open('mantis.jpg'),PretrainedModel(model,layer_name, is_sequential= True)) # for facenet sequential is not true
plt.imshow(image)
