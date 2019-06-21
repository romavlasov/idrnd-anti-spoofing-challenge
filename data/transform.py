import cv2
import torch

import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image= t(image)
        return image


class OneOf(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image):
        transform = np.random.choice(self.transforms)
        image = transform(image)
        return image
    
    
class RandomApply(object):
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob    
        
    def __call__(self, image):
        for t in self.transforms:
            if np.random.rand() < self.prob:
                image = t(image)
        return image


class RandomApply(object):
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob
        
    def __call__(self, image):
        for t in self.transforms:
            if np.random.rand() < self.prob:
                image = t(image)
        return image


class CenterCrop(object):
    def __init__(self, size=None):
        self.size = size
        
    def __call__(self, image):
        height, width = image.shape[:2]
        
        if height > width:
            center = height // 2
            top = center - width // 2
            bottom = center + width // 2
            image = image[top:bottom, :]
        else:
            center = width // 2
            left = center - height // 2
            right = center + height // 2
            image = image[:, left:right]
        
        if self.size and self.size < image.shape[0]:
            center = height // 2
            top = center - self.size // 2
            bottom = center + self.size // 2
            left = center - self.size // 2
            right = center + self.size // 2
            image = image[top:bottom, left:right]
            
        return image
    

class RandomCrop(object):
    def __init__(self, ratio):
        self.ratio = ratio
        
    def __call__(self, image):
        width = int(image.shape[1] * self.ratio)
        height = int(image.shape[0] * self.ratio)
        
        min_x = image.shape[1] - width
        min_y = image.shape[0] - height
        
        x = np.random.randint(0, min_x) if min_x else 0
        y = np.random.randint(0, min_y) if min_y else 0
        
        image = image[y:y + height, x:x + width]
        return image


class Contrast(object):
    def __init__(self, lower=0.9, upper=1.1):
        self.lower = lower
        self.upper = upper

    def __call__(self, image):
        alpha = np.random.uniform(self.lower, self.upper)
        image *= alpha
        image = np.clip(image, 0, 1)
        return image


class Brightness(object):
    def __init__(self, delta=0.125):
        self.delta = delta

    def __call__(self, image):
        delta = np.random.uniform(-self.delta, self.delta)
        image += delta
        image = np.clip(image, 0, 1)
        return image
    

class GaussianBlur(object):
    def __init__(self, kernel=3):
        self.kernel = (kernel, kernel)
    
    def __call__(self, image):
        image = cv2.blur(image, self.kernel)
        return image


class Expand(object):
    def __init__(self, size=1024, diff=0.3, noise=False):
        self.size = size
        self.noise = noise
        self.diff = diff

    def __call__(self, image):
        height, width = image.shape[:2]
        max_ratio = self.size / max(height, width)
        min_ratio = max_ratio * self.diff

        ratio = np.random.uniform(min_ratio, max_ratio)
        left = np.random.uniform(0, self.size - width*ratio)
        top = np.random.uniform(0, self.size - height*ratio)

        expand_image = np.zeros((self.size, self.size, 3), dtype=image.dtype)
        if self.noise:
            mean = np.full(3, 0.5)
            std = np.full(3, 0.5)
            expand_image = cv2.randn(expand_image, mean, std)
        expand_image = np.clip(expand_image, 0, 1)
        
        image = cv2.resize(image, (int(width*ratio), int(height*ratio)))
        
        expand_image[int(top):int(top) + int(height*ratio),
                     int(left):int(left) + int(width*ratio)] = image
        image = expand_image

        return image


class Pad(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):
        height, width = image.shape[:2]
        
        ratio = self.size / max(height, width)
        
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        
        # new_size should be in (width, height) format
        
        image = cv2.resize(image, (new_width, new_height))
        
        delta_w = self.size - new_width
        delta_h = self.size - new_height
        
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        color = [0, 0, 0]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        return image
    
    
class Rotate(object):
    def __init__(self, angle=10, aligne=False):
        self.angle = angle
        self.aligne = aligne
        
    def __call__(self, image):        
        angle = np.random.uniform(-self.angle, self.angle)

        height, width = image.shape[:2]
        cX, cY = width / 2, height / 2
     
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

        if self.aligne:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
     
            width = int((height * sin) + (width * cos))
            height = int((height * cos) + (width * sin))
     
            M[0, 2] += (width / 2) - cX
            M[1, 2] += (height / 2) - cY
            
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT)
        return image
    

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        height, width = image.shape[:2]
        
        h_scale = self.size / height
        w_scale = self.size / width
        
        image = cv2.resize(image, (self.size, self.size))
        return image


class HorizontalFlip(object):
    def __call__(self, image):
        image = cv2.flip(image, 1)
        return image


class ToTensor(object):
    def __call__(self, image, target=None, mask=None):
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image.float()


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean or [0.485, 0.456, 0.406])
        self.std = np.array(std or [0.229, 0.224, 0.225])

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


class Transforms(object):
    def __init__(self, input_size, train=True):
        self.train = train

        self.transforms_train = RandomApply([
            RandomCrop(0.9),
            Rotate(angle=10, aligne=False),
            HorizontalFlip(),
        ])

        self.transforms_test = RandomApply([
            HorizontalFlip(),
        ])

        self.normalize = Compose([
            Resize(input_size),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, image):
        if self.train:
            image = self.transforms_train(image)
        else:
            image = self.transforms_test(image)
            
        image = self.normalize(image)
        return image
