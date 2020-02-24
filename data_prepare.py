from __future__ import print_function
import zipfile
import os

import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.utils.data as data
import torch
from torchvision.datasets.utils import download_url, list_dir, list_files

from imgaug import augmenters as iaa
import numpy as np
import scipy.io
from os.path import join

IMG_SIZE = 32 
VAL_RATE = 0.2

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.SomeOf((1, None), [
                iaa.Multiply((1, 1.2)),
                iaa.Sharpen(alpha=(0.0, 0.75), lightness=(1, 1.5)),
                iaa.Affine(shear=(-20, 20)),
                iaa.Affine(rotate=(-20, 20)),
                iaa.ContrastNormalization((0.5, 1.5)),
            ], random_order=True))
        ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

gtsrb_transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

gtsrb_transform_train = transforms.Compose([
    ImgAugTransform(),
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def gtsrb_initialize_data(folder):
    val_size = 0
    train_folder = folder + '/GTSRB/Training'
    val_folder = folder + '/GTSRB/Test'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                image_num = int(len(os.listdir(train_folder + '/' + dirs)) / 30)
                for f in os.listdir(train_folder + '/' + dirs):
                    for i in range(int(image_num * VAL_RATE)):
                        if f.startswith(format(i, '05d')): 
                            # move file to validation folder
                            os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
                            val_size += 1
        print("val_size = " + str(val_size))

