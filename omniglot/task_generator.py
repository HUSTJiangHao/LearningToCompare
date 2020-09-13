# 此文件用于生成训练任务

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler



class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


# 获取所有的文件夹名称，每一个文件夹里面包含同一类,生成文件夹路径的list
def omniglot_character_folders():
    data_folder = '../datas/omniglot_resized/'


    character_folders = [os.path.join(data_folder, family, character).replace("\\", "/") \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)  # 随机打乱

    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]  # 前1200个作为训练集 from 论文
    metaval_character_folders = character_folders[num_train:]  # 剩余的作为验证
    return metatrain_character_folders, metaval_character_folders


# 一个训练/测试任务  包含训练集S 5个不同类型的样本 和Q 19*5个样本 中样例的路径 以及对应的标签
class OmniglotTask(object):
    def __init__(self, character_folders, num_classes, train_num,test_num):

        self.character_folders = character_folders
        self.num_classes = num_classes  # 5way
        self.train_num = train_num  # 1shot
        self.test_num = test_num  # 19

        class_folders = random.sample(self.character_folders,self.num_classes)  # 随机选取5个种类的文件夹返回
        labels = np.array(range(len(class_folders)))  # 以 0,1,2,3，4作为标签
        labels = dict(zip(class_folders, labels))
        samples = dict()  # key : 文件夹名称 value: 文件夹下所有文件名称的list

        self.train_roots = []
        self.test_roots = []

        for c in class_folders:  # class_folders： 选取的5个种类的文件夹名称的list

            temp = [os.path.join(c, x) for x in os.listdir(c)]  # 当前种类的所有文件/样本名称/路径 的list
            samples[c] = random.sample(temp, len(temp))  # 随机打乱

            self.train_roots += samples[c][:train_num]  # 第一个作为样本集 S
            self.test_roots += samples[c][train_num:train_num+test_num] # 其他19个作为查询集Q

        # for x in self.train_roots:
        #     print(labels[self.get_class(x)])

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots] # 获取标签
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        # print(*sample.split('\\'))
        # print(os.path.join(*sample.split('\\')[:-1]))
        return os.path.join(*sample.split('\\')[:-1])


# 从一个task中 创建dataset
class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


# dataset
class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')  # 灰度图像
        image = image.resize((28,28), resample=Image.LANCZOS)


        if self.transform is not None:
            image = self.transform(image)
            print("   ")
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)


        return image, label


# 自定义采样方式
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class  # 1shot /19
        self.num_cl = num_cl # 5shot
        self.num_inst = num_inst # 每类个数S 1 Q 19
        self.shuffle = shuffle

    def __iter__(self):
  # 在 num_inst个中选出 num_per_class个样本
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train',shuffle=True,rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    # normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    normalize = transforms.Normalize(mean=[ 0.92206], std=[ 0.08426])
    dataset = Omniglot(task,split=split,transform=transforms.Compose([Rotate(rotation),transforms.ToTensor(),normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader