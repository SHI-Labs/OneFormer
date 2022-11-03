# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/colormap.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

"""
An awesome colormap for really neat visualizations.
Copied from Detectron, and removed gray colors.
"""

import numpy as np
import random
random.seed(0)

__all__ = ["colormap", "random_color", "random_colors"]

_COLORS = []

def gen_color():
    color = tuple(np.round(np.random.choice(range(256), size=3)/255, 3))
    if color not in _COLORS and np.mean(color) != 0.0:
        _COLORS.append(color)
    else:
        gen_color()


for _ in range(300):
    gen_color()


def colormap(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def random_colors(N, rgb=False, maximum=255):
    """
    Args:
        N (int): number of unique colors needed
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a list of random_color
    """
    indices = random.sample(range(len(_COLORS)), N)
    ret = [_COLORS[i] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret


if __name__ == "__main__":
    import cv2

    size = 100
    H, W = 10, 10
    canvas = np.random.rand(H * size, W * size, 3).astype("float32")
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = _COLORS[idx]
    cv2.imshow("a", canvas)
    cv2.waitKey(0)