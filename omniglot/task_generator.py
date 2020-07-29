# 此文件用于生成训练任务

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
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


# 一个训练/测试任务
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