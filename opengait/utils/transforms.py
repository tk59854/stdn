import torch
import random
import math
import numpy as np
import cv2

import imgaug.augmenters as iaa


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, imgs):

        if random.uniform(0, 1) >= self.probability:
            return imgs
        for attempt in range(100):
            area = imgs.shape[1] * imgs.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < imgs.shape[2] and h < imgs.shape[1]:
                x1 = random.randint(0, imgs.shape[1] - h)
                y1 = random.randint(0, imgs.shape[2] - w)

                imgs[:, x1:x1 + h, y1:y1 + w] = self.mean[0]

                return imgs

        return imgs


class SilBlur:
    def __init__(self, kernel=(3, 3)):
        self.kernel = kernel

    def __call__(self, sil):
        return cv2.blur(sil, self.kernel)


class SilDropout:
    def __init__(self, p=0.2):
        self.aug = iaa.Dropout(p=p)

    def __call__(self, sil):
        return iaa_wrapper(sil, self.aug)


class SilCutout:

    def __init__(self, **kwargs):
        self.aug = iaa.Cutout(**kwargs)

    def __call__(self, sil):
        return iaa_wrapper(sil, self.aug)


class Cutout(object):
    def __init__(self, n_holes=2, rate=(0.3, 0.2)):
        self.n_holes = n_holes
        self.rate = rate

    def __call__(self, img):
        h = img.shape[1]
        w = img.shape[2]
        length_h = int(h * self.rate[0])
        length_w = int(w * self.rate[1])

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length_h // 2, 0, h)
            y2 = np.clip(y + length_h // 2, 0, h)
            x1 = np.clip(x - length_w // 2, 0, w)
            x2 = np.clip(x + length_w // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        img = img * mask

        return img


class SilAffine:
    def __init__(self, **kwargs):
        self.aug = iaa.Affine(**kwargs)

    def __call__(self, sil):
        return iaa_wrapper(sil, self.aug)


def iaa_wrapper(sil, aug):  # iaa的灰度图需要shape=(n, h, w, 1)
    sil = np.expand_dims(sil, -1)
    sil = aug(images=sil)
    sil = sil.squeeze(-1)
    return sil

