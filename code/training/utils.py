import csv
import random
import torch
import os
import numpy as np
import torch.nn as nn
import torch.utils.data
from PIL import ImageFile
import time as sys_time
import sys

from PIL import Image
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class fully_connected(nn.Module):
    def __init__(self, model, num_ftrs, num_classes):  
        super(fully_connected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        out_3 = torch.relu(out_3)
        return out_1, out_3


class Logger(object):
    def __init__(self, stream=sys.stdout, task_name=None):
        output_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "log")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(sys_time.strftime('%Y-%m-%d-%H-%M')+"-" + task_name)
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


class PTC(torch.utils.data.Dataset):
    def __init__(self, root, slide='11_breast_cancer3',transform=None, stain_norm=False):
        super(PTC, self).__init__()
        self.root = root
        self.slide = slide
        self.transform = transform
        self.stain_norm = stain_norm
        self.patch =[]
        # self.class_num = list(0. for i in range(clusters))
        patch_list = {}
        patch_path = os.path.join(root , slide , 'patches')
        patch = os.listdir(patch_path)
        for i in range(len(patch)):
            patch_list[slide+'_'+patch[i].split('.')[0]] = patch[i]

        self.label = []

        label_path = os.path.join(root, slide, 'high_250_stdata.csv')
        label_file = open(label_path, 'r')
        csv_reader = csv.reader(label_file)
        for row in csv_reader:
            id = row[0]
            if id == 'id':
                continue
            pca_vector = row[1:]
            pca_vector = list(map(float, pca_vector))

            id = slide + '_' + id

            try:
                self.patch.append(patch_list[id])
                self.label.append(pca_vector)
            except KeyError:
                continue
            # self.class_num[int(clus)-1] += 1
        label_file.close()

    def __getitem__(self, index):
        patch_path = os.path.join(self.root, self.slide, 'patches', self.patch[index])
        patch_file = open(patch_path, 'rb')
        name = self.patch[index]
        label = []
        label.append(self.label[index])
        label = torch.Tensor(label)
        patch = Image.open(patch_file).convert('RGB')
        data = transforms.Resize((224, 224))(patch)
        patch_file.close()
        if self.transform is not None:
            data = self.transform(data)
        return name, data, label

    def __len__(self):
        return len(self.patch)


class PTC_2x(torch.utils.data.Dataset):
    def __init__(self, root, slide='11_breast_cancer3',transform=None, stain_norm=False):
        super(PTC_2x, self).__init__()
        self.root = root
        self.slide = slide
        self.transform = transform
        self.stain_norm = stain_norm
        self.patch =[]
        # self.class_num = list(0. for i in range(clusters))
        patch_list = {}
        patch_path = os.path.join(root , slide , '2xpatches')
        patch = os.listdir(patch_path)
        for i in range(len(patch)):
            patch_list[slide+'_'+patch[i].split('.')[0]] = patch[i]

        self.patch = patch

    def __getitem__(self, index):
        patch_path = os.path.join(self.root, self.slide, '2xpatches', self.patch[index])
        patch_file = open(patch_path, 'rb')
        name = self.patch[index]
        # label = []
        # label.append(self.label[index])
        label = torch.zeros(250)
        patch = Image.open(patch_file).convert('RGB')
        data = transforms.Resize((224, 224))(patch)
        patch_file.close()
        if self.transform is not None:
            data = self.transform(data)
        return name, data, label

    def __len__(self):
        return len(self.patch)


class PTC_2000(torch.utils.data.Dataset):
    def __init__(self, root, slide='11_breast_cancer3',transform=None, stain_norm=False):
        super(PTC_2000, self).__init__()
        self.root = root
        self.slide = slide
        self.transform = transform
        self.stain_norm = stain_norm
        self.patch =[]
        # self.class_num = list(0. for i in range(clusters))
        patch_list = {}
        patch_path = os.path.join(root , slide , 'patches')
        patch = os.listdir(patch_path)
        for i in range(len(patch)):
            patch_list[slide+'_'+patch[i].split('.')[0]] = patch[i]

        self.label = []

        label_path = os.path.join(root, slide, 'high_2000_stdata.csv')
        label_file = open(label_path, 'r')
        csv_reader = csv.reader(label_file)
        for row in csv_reader:
            id = row[0]
            if id == 'id':
                continue
            pca_vector = row[1:]
            pca_vector = list(map(float, pca_vector))

            id = slide + '_' + id

            try:
                self.patch.append(patch_list[id])
                self.label.append(pca_vector)
            except KeyError:
                continue
            # self.class_num[int(clus)-1] += 1
        label_file.close()

    def __getitem__(self, index):
        patch_path = os.path.join(self.root, self.slide, 'patches', self.patch[index])
        patch_file = open(patch_path, 'rb')
        name = self.patch[index]
        label = []
        label.append(self.label[index])
        label = torch.Tensor(label)
        patch = Image.open(patch_file).convert('RGB')
        data = transforms.Resize((224, 224))(patch)
        patch_file.close()
        if self.transform is not None:
            data = self.transform(data)
        return name, data, label

    def __len__(self):
        return len(self.patch)