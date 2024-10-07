import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F

#Image processing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numbers

from tqdm import tqdm
import math

#Utility functions
#transform=models.VGG16_Weights.IMAGENET1K_V1.transforms()
IMAGENET_MEAN=torch.tensor([0.485, 0.456, 0.406],dtype=torch.float32).view(3,1,1) # we unsqueeze to support broadcasting. 3,H,W/ 3,1,1
IMAGENET_STD =torch.tensor([0.229, 0.224, 0.225],dtype=torch.float32).view(3,1,1 ) # we unsqueeze to support broadcasting. 3,H,W/ 3,1,1

def pre_prcoess_PIL_to_np_to_tensor(PIL_image):
  image= transforms.ToTensor()(PIL_image) # This moves image from [0,255] range to [0,1 ]range

  if image.ndim==4:

    IMAGENET_MEAN1 = IMAGENET_MEAN.unsqueeze(dim=0)
    IMAGENET_STD1 = IMAGENET_STD.unsqueeze(dim=0)
    image=(image-IMAGENET_MEAN1)/IMAGENET_STD1

  elif image.ndim==3:
    image=(image-IMAGENET_MEAN)/IMAGENET_STD
    image=image.unsqueeze(0)

  else:
    raise ValueError(f' image dimensions are invalid')


  return image


def pyramid_ratio_generator(tensor_image,number_of_pyramids,pyramid_ratio):
    threshold_size = 10  # Minimum acceptable size for the pyramid levels
    original_shape = tensor_image.shape[-2:]  # Assuming the input is (...., H, W)

    image_sizes = []
    for pyramid_number in range(number_of_pyramids):
      scaling_factor = pyramid_ratio ** (torch.tensor(pyramid_number - number_of_pyramids + 1) )
      new_size = tuple(int(scaling_factor * dim) for dim in original_shape)
      image_sizes.append(new_size)

      if any(dim < threshold_size  for dim in new_size ):
        raise ValueError(f'Pyramid level {pyramid_number + 1} with pyramid ratio {pyramid_ratio} '
                          f'results in too small pyramid levels (size={new_size}). '
                            'Please adjust the parameters.')

    return image_sizes

def post_process_torch_to_numpy(tensor_image):

  if tensor_image.shape[0]==1: # if only there is one image
    tensor_image=tensor_image[0]
    tensor_image= (tensor_image * IMAGENET_STD) + IMAGENET_MEAN  # de-normalize
    numpy_image=np.moveaxis(tensor_image.detach().cpu().numpy(), 0, 2)
    return (np.clip(numpy_image, 0., 1.) * 255).astype(np.uint8)

  else:
    IMAGENET_MEAN1 = IMAGENET_MEAN.unsqueeze(dim=0)
    IMAGENET_STD1 = IMAGENET_STD.unsqueeze(dim=0)

    image=(tensor_image * IMAGENET_STD1)+IMAGENET_MEAN1

    numpy_image=np.moveaxis(image.detach().numpy(), 1, 3)
    return (np.clip(numpy_image, 0., 1.) * 255).astype(np.uint8)

class CascadeGaussianSmoothing(nn.Module):
    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers, hardcoded to use 3 different Gaussian kernels
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.to(DEVICE)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3
