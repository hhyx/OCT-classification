import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import time


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


class Preproc(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        w, h = sample.size                         # 获得图片的宽和高
        sample_numpy = np.array(sample)            # 获得每个像素的rpg值
        mean = np.mean(sample_numpy)               # 求像素的平均值
        std = np.std(sample_numpy)                 # 求像素的标准差
        threshold = mean + std*self.sigma

        top_index = 0
        for index in range(int(h/2)):
            if np.mean(sample_numpy[index, :, 0]) > threshold:
                top_index = index + 1
            else:
                break

        bottom_index = h-1
        for index in range(h-1, int(h/2), -1):
            if np.mean(sample_numpy[index, :, 0]) > threshold:
                bottom_index = index - 1
            else:
                break

        left_index = 0
        for index in range(int(w/2)):
            if np.mean(sample_numpy[:, index, 0]) > threshold:
                left_index = index + 1
            else:
                break

        right_index = w - 1
        for index in range(w - 1, int(w/2), -1):
            if np.mean(sample_numpy[:, index, 0]) > threshold:
                right_index = index - 1
            else:
                break

        sample_numpy = sample_numpy[top_index:bottom_index+1, left_index:right_index+1]
        return Image.fromarray(sample_numpy)


class Rescale(object):     # 利用双三次插值算法将图片所辖
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
    
        sample = sample.resize((new_h, new_w), Image.BICUBIC)
    
        return sample


class Resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        new_h, new_w = int(self.output_size), int(self.output_size)

        sample_numpy = np.array(sample)
        ticks = time.time()
        image = Image.fromarray(sample_numpy)
        image.save(str(int(ticks)) + '_0001.png')

        sample = sample.resize((new_h, new_w), Image.BICUBIC)

        sample_numpy = np.array(sample)
        image = Image.fromarray(sample_numpy)
        image.save(str(int(ticks)) + '_0002.png')

        return sample


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.size
        new_h, new_w = self.output_size

        if h == new_h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        if w == new_w:
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        sample = sample.crop((top, left, top + new_h, left + new_w))

        return sample


class ToTensor(object):

    def __call__(self, sample):
        input_image = np.array(sample, np.float32) / 255.0

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input_image = input_image.transpose((2, 0, 1))
        return torch.from_numpy(input_image)


class Normalization(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample):
        sample = (sample - self.mean) / self.std

        return sample
