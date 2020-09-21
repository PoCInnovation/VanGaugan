import torch
import csv
import string
from PIL import Image
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
from enum import Enum

class CelebaDataset(Dataset) :
    def __init__(self, attr_path="./dataset/CelebA/list_attr_celeba.csv",img_dir="./dataset/CelebA/images", transform=None):
        self.img_dir = Path(img_dir)
        self.attr_path = attr_path
        self.transform = transform
        self.files_it = iter(os.listdir(img_dir))
        self.length, self.attr_names, self.attr_it = self.process_attr_file(attr_path)
    
    def __len__(self):
        return (self.length)

    def __getitem__(self, idx):
        try :
            f = next(self.files_it)
        except StopIteration : # return None object if there are no longer images to load
            return (None, None)
        with Image.open(self.img_dir / f) as image :
            image = self.transform(image) if self.transform is not None else image
            data = np.asarray(image)
        elem_attr = next(self.attr_it)
        # 1 = man, 2 = woman
        sex = int(elem_attr[21]) if elem_attr[21] == 1 else 0
        return (data, torch.tensor(sex))

    # Get len, attributes name, and a pointer on attributes matrices iterator from file
    def process_attr_file(self, filepath) :
        attr_list = []
        with open(filepath, "r") as csvfile :
            my_reader = csv.reader(csvfile)
            size = int(next(my_reader)[0])
            attr_names = next(my_reader)
            for line in my_reader :
                matrice = [int(attr) for attr in line[1:]]
                attr_list.append(matrice)
        return size, attr_names[1:], iter(attr_list)