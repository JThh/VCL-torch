import numpy as np
from torchvision.transforms import ToTensor
import torch


class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()


class Scale(object):
    """Scale images down to have [0,1] float pixel values"""
    def __init__(self, max_value=255):
        self.max_value = max_value

    def __call__(self, sample):
        return sample / self.max_value


class Permute(object):
    """ Apply a fixed permutation to the pixels in the image. """
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, sample):
        return sample[self.permutation]


class Permute2D:
    def __init__(self, perm):
        """
        Args:
            perm (Tensor): The permutation to apply to the 2D images.
        """
        self.perm = perm

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be permuted.

        Returns:
            Tensor: Permuted image.
        """
        if not isinstance(img, torch.Tensor):
            img = ToTensor()(img)  # Convert PIL image to tensor
        C, H, W = img.shape
        img = img.view(C, -1)  # Flatten spatial dimensions
        img = img[:, self.perm]  # Apply permutation
        img = img.view(C, H, W)  # Reshape back to original spatial dimensions
        return img
