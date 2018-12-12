import os
import numpy as np
import torch
from torchvision import datasets
from skimage import io, transform
from skimage.color import rgb2gray
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):

        self.img_list = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):


        img_name = os.path.join(self.root_dir, self.img_list[idx])
        image = transform.resize(io.imread(img_name), (256, 256), mode='constant')
        gray_img = transform.resize(rgb2gray(image), (1, 256, 256))
        image = np.transpose(image, (2, 0, 1))


        sample = (gray_img, image)

        return sample