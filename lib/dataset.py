# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-19
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MLDataset(Dataset):
    def __init__(self, data_path, label_path, transform):
        super(MLDataset, self).__init__()

        self.labels = [line.strip() for line in open(label_path)]
        self.num_classes = len(self.labels)
        self.label2id = {label:i for i, label in enumerate(self.labels)}

        self.data = []
        with open(data_path, 'r') as fr:
            for line in fr.readlines():
                image_path, image_label = line.strip().split('\t')
                image_label = [self.label2id[l] for l in image_label.split(',')]
                self.data.append([image_path, image_label])
        self.transform = transform

    def __getitem__(self, index):
        image_path, image_label = self.data[index]
        image_data = Image.open(image_path).convert('RGB')
        x = self.transform(image_data)

        # one-hot encoding for label
        y = np.zeros(self.num_classes).astype(np.float32)
        y[image_label] = 1.0
        return x, y
        
    def __len__(self):
        return len(self.data)
