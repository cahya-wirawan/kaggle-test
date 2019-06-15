import numpy as np
import scipy.misc
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from config import INPUT_SIZE
import Augmentor


class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class StanfordCars(Dataset):
    """
    # Description:
        Basic class for retrieving images and labels

    # Member Functions:
        __init__(self, phase, shape):   initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            shape:                      output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, datapath, phase='train', shape=INPUT_SIZE):
        assert phase in ['train', 'val', 'test']
        self.datapath = datapath
        self.phase = phase
        self.files = (datapath/phase).glob("**/*.jpg")
        carnames = datapath/"../../names.csv"
        df = pd.read_csv(carnames, header=None)
        car_labels = {name.replace('C/V', 'C-V'): i for i,name in enumerate(list(df[0]))}
        self.data_list = []

        for fullpath in self.files:
            filepath = str(fullpath).split('/')
            self.data_list.append((fullpath, car_labels[filepath[-2]]))

        self.shape = shape

        aug = Augmentor.Pipeline()
        #aug.crop_random(probability=1, percentage_area=0.9, randomise_percentage_area=True)
        aug.resize(probability=1.0, width=300, height=300, resample_filter='BICUBIC')
        aug.rotate(probability=0.9, max_left_rotation=25, max_right_rotation=25)
        aug.flip_left_right(probability=0.5)
        #aug.flip_top_bottom(probability=0.2)
        #aug.skew_tilt(probability=0.5)
        #aug.random_distortion(probability=0.5, grid_width=8, grid_height=8, magnitude=4)
        aug.random_brightness(probability=0.75, min_factor=0.8, max_factor=1.2)
        aug.random_contrast(probability=0.75, min_factor=0.8, max_factor=1.2)
        #p.random_color(probability=1.0, min_factor=0.2, max_factor=1.0)
        aug.zoom(probability=0.75, min_factor=1.0, max_factor=1.1)
        aug.resize(probability=1.0, width=self.shape[0], height=self.shape[1], resample_filter='BICUBIC')

        self.transforms = {
            'train': transforms.Compose([
                aug.torch_transform(),
                #transforms.RandomAffine(90, translate=None, scale=None, shear=None,
                #                        resample=False, fillcolor=0),
                #transforms.Resize(size=(300, 300)),
                #transforms.RandomCrop(INPUT_SIZE),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(size=(300, 300)),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=(300, 300)),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def __getitem__(self, item):
        image = Image.open(self.data_list[item][0]).convert('RGB')  # (C, H, W)
        image = self.transforms[self.phase](image)
        assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]

        if True or self.phase != 'test':
            # filename of image should have 'id_label.jpg/png' form
            label = self.data_list[item][1]  # label
            return image, label
        else:
            # filename of image should have 'id.jpg/png' form, and simply return filename in case of 'test'
            return image, str(self.data_list[item][0])

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    dataset = CUB(root='./CUB_200_2011')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(root='./CUB_200_2011', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
