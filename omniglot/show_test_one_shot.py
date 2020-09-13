import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random
import scipy as sp
import scipy.stats
from PIL import Image

# Hyper Parameters
FEATURE_DIM = 64  # 论文使用64层提取特征
RELATION_DIM = 8  # 输出得分最后一层为8
CLASS_NUM = 5  # 5way
SAMPLE_NUM_PER_CLASS = 1  # 1shot
BATCH_NUM_PER_CLASS = 19  # 其余19张作为查询集 Q
EPISODE = 1000000
TEST_EPISODE = 1000
LEARNING_RATE = 0.001
GPU = 0
TEST_NUMS = 5
# HIDDEN_UNIT = 10

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a),scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)


    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    image_folders = []  # 每个文件夹包含一种字符
    folderIndexs = np.random.randint(0, len(metatest_character_folders), 5)
    for i in folderIndexs:
        image_folders.append(metatest_character_folders[i])


    train_image_root = [os.path.join(fold,  os.listdir(fold)[i]).replace("\\", "/") for fold in image_folders for i in np.random.randint(0,19,1)]
    query_image_root = [os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for fold in image_folders for i in range(20)]

    query_image_root = list(set(query_image_root) - set(train_image_root))

    query = [query_image_root[i] for i in np.random.randint(0,94,TEST_NUMS)]
    query_image_root = query

    sample_images = torch.zeros(5,1,28,28)
    test_images = torch.zeros(TEST_NUMS,1,28,28)

    for i in range(5):
        image = Image.open(train_image_root[i])
        image = image.convert('L')  # 灰度图像
        image = image.resize((28, 28), resample=Image.LANCZOS)

        normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        image = transform(image)

        sample_images[i] = image

    for i in range(TEST_NUMS):
        image = Image.open(query_image_root[i])
        image = image.convert('L')  # 灰度图像
        image = image.resize((28, 28), resample=Image.LANCZOS)

        normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        image = transform(image)

        test_images[i] = image



    # calculate features
    sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
    test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

    # calculate relations
    # each batch sample link to every samples to calculate relations
    # to form a 100x128 matrix for relation network
    # sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
    sample_features_ext = sample_features.unsqueeze(0).repeat(TEST_NUMS, 1, 1, 1, 1)
    test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
    test_features_ext = torch.transpose(test_features_ext,0,1)

    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,5,5)
    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

    _,predict_labels = torch.max(relations.data,1)
    print(" ")









if __name__ == '__main__':
    main()
