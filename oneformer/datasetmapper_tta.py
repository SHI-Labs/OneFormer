import copy
import numpy as np
from typing import List
import torch
from fvcore.transforms import NoOpTransform
from torch import nn

from detectron2.config import configurable
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    apply_augmentations,
)

__all__ = ["DatasetMapperTTA"]


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    @configurable
    def __init__(self, min_sizes: List[int], max_size: int, flip: bool):
        """
        Args:
            min_sizes: list of short-edge size to resize the image to
            max_size: maximum height or width of resized images
            flip: whether to apply flipping augmentation
        """
        self.min_sizes = min_sizes
        self.max_size = max_size
        self.flip = flip

    @classmethod
    def from_config(cls, cfg):
        return {
            "min_sizes": cfg.TEST.AUG.MIN_SIZES,
            "max_size": cfg.TEST.AUG.MAX_SIZE,
            "flip": cfg.TEST.AUG.FLIP,
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.
        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdge(min_size, self.max_size)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                aug_candidates.append([resize, flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret