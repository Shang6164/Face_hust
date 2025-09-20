import numpy as np
import imageio
import os
from sklearn import preprocessing
import torch
import cv2
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys

sys.path.append("..")


class DataLoader(object):
    def __init__(self, root, flip=True):
        self.image_list = []
        self.label_name = []
        self.flip = flip
        for r, _, files in os.walk(root):
            for f in files:
                self.image_list.append(os.path.join(r, f))
                self.label_name.append(os.path.basename(r))
        le = preprocessing.LabelEncoder()
        self.label_list = le.fit_transform(self.label_name)
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = imageio.imread(img_path)
        img = cv2.resize(img, (112, 112))

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    data_dir = 'C:/Users/daoda/OneDrive/Documents/Anh do an'
    dataset = DataLoader(root=data_dir)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
