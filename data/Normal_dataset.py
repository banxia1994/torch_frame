import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
from PIL import Image
import numpy as np
from utils import cv_utils


# read data from txt
class Normal_dataset(DatasetBase):
    def __init__(self,opt,is_for_train):
        super(Normal_dataset,self).__init__(opt,is_for_train)
        self._name = 'CurrentDataset'

        # read dataset
        self._read_data_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        im_path = (self.all_file[index])[0]
        label = (self.all_file[index])[1]
        im = cv_utils.read_cv2_img(im_path)

        img = self._transform(Image.fromarray(im))

        # pack
        sample = {
            'img':img,
            'label':label
        }

        return sample
    def _create_transform(self):
        if self._is_for_train:
            transform_list = [transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std = [0.5, 0.5, 0.5]),
                              ]
        self._transform = transforms.Compose(transform_list)
    def __len___(self):
        return self._dataset_size

    def _read_data_paths(self):
        # read data from txt, simple classification
        if os.path.isfile(self.root):
            all_file = []
            with open (self.root,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    all_file.append((line[0],int(line[1])))
            self.all_file = all_file
            self._dataset_size = len(self.all_file)

        #self.root = self._opt.data_dir
        else:
            assert False,'error to load data form txt'