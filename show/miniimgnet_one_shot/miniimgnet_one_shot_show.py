import PySide2
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from PySide2 import QtGui
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy as sp
import scipy.stats
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
# import task_generator as tg
import os
import math
import miniimgnet_one_shot_model as this_model

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

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


class Show_widget:

    def __init__(self):
        qfile_stats = QFile('miniimgnet_one_shot.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        self.ui = QUiLoader().load(qfile_stats)

        self.image_folders = []
        self.encode_path = ''
        self.relation_path = ''
        self.query_image_root = []  # 查询集路径
        self.train_image_root = []
        self.feature_encoder = []
        self.feature_encoder = this_model.CNNEncoder()
        self.relation_network = this_model.RelationNetwork(FEATURE_DIM, RELATION_DIM)
        self.sample_images = torch.zeros(1)
        self.test_images = torch.zeros(1)
        self.random_indexs = []
        self.myclass = dict()

        self.ui.button_select_folds.clicked.connect(self.on_select_folders)
        self.ui.button_load_model.clicked.connect(self.on_load_model)
        self.ui.button_multi_show.clicked.connect(self.on_batch_show)
        self.ui.button_calc.clicked.connect(self.on_calc)
        self.ui.button_rechoose.clicked.connect(self.on_rechoose)

        self.pix_train_images = [self.ui.train_img1, self.ui.train_img2, self.ui.train_img3, self.ui.train_img4,
                                 self.ui.train_img5]
        self.pix_query_images = [[self.ui.img11, self.ui.img12, self.ui.img13, self.ui.img14, self.ui.img15],
                                 [self.ui.img21, self.ui.img22, self.ui.img23, self.ui.img24, self.ui.img25],
                                 [self.ui.img31, self.ui.img32, self.ui.img33, self.ui.img34, self.ui.img35]]

    def on_select_folders(self):

        self.myclass = dict()
        self.image_folders = []

        dialog = QFileDialog(directory="../../datas/miniimagenet/_PNG")
        dialog.setFileMode(QFileDialog.Directory)


        for i in range(5):
            if dialog.exec_():
                fileName = dialog.selectedFiles()
                self.image_folders.append(fileName[0])
                self.myclass[fileName[0].split("/")[-1]] = i

                self.ui.TextEdit.appendPlainText(
                    " class：" + str(i) + "  " + fileName[0].split("/")[-2] + '/' + fileName[0].split("/")[-1])


        train_image_root = [os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for fold in self.image_folders
                            for i
                            in np.random.randint(0, 600, 1)]
        query_image_root = [os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for fold in self.image_folders
                            for i
                            in range(600)]

        query_image_root = list(set(query_image_root) - set(train_image_root))
        # query = [query_image_root[i] for i in np.random.randint(0, 94, TEST_NUMS)]

        self.query_image_root = query_image_root
        self.train_image_root = train_image_root


        for i in range(5):
            image = QtGui.QPixmap(train_image_root[i]).scaled(self.ui.train_img1.width(), self.ui.train_img1.height())
            self.pix_train_images[i].setPixmap(image)

    def on_load_model(self):
        dialog = QFileDialog(caption="选择编码网络", directory="../models/")
        if dialog.exec_():
            fileName = dialog.selectedFiles()
            self.encode_path = fileName[0]
            self.ui.TextEdit.appendPlainText("编码网络模型：" + fileName[0])

        dialog = QFileDialog(caption="选择关系网络", directory="../models/")
        if dialog.exec_():
            fileName = dialog.selectedFiles()
            self.relation_path = fileName[0]
            self.ui.TextEdit.appendPlainText("关系网络模型：" + fileName[0])

        self.ui.TextEdit.appendPlainText("--加载网络模型--")

        self.feature_encoder.cuda(GPU)
        self.relation_network.cuda(GPU)

        if os.path.exists(self.encode_path):
            self.feature_encoder.load_state_dict(torch.load(self.encode_path))
            self.ui.TextEdit.appendPlainText("--成功加载编码模型--")

        if os.path.exists(self.relation_path):
            self.relation_network.load_state_dict(torch.load(self.relation_path))
            self.ui.TextEdit.appendPlainText("--成功加载关系模型--")

    def on_batch_show(self):

        for i in range(5):
            self.pix_query_images[0][i].setPixmap(QPixmap(""))
            self.pix_query_images[1][i].setPixmap(QPixmap(""))
            self.pix_query_images[2][i].setPixmap(QPixmap(""))

        self.sample_images = torch.zeros(5, 3, 84, 84)
        self.test_images = torch.zeros(TEST_NUMS, 3, 84, 84)
        for i in range(5):

            image = Image.open(self.train_image_root[i])
            image = image.convert('RGB')
            normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            image = transform(image)
            self.sample_images[i] = image

        self.random_indexs = np.random.randint(0, len(self.query_image_root) - 1, 5)

        batch_class = []
        for i in range(5):
            batch_class.append(self.query_image_root[self.random_indexs[i]].split("/")[-2])
            image = Image.open(self.query_image_root[self.random_indexs[i]])
            image = image.convert('RGB')
            normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            image = transform(image)

            self.test_images[i] = image

        self.ui.TextEdit.appendPlainText("sample: " + str(self.myclass[batch_class[0]]) + "  " + str(self.myclass[batch_class[1]]) + "  " +
                                         str(self.myclass[batch_class[2]]) + "  " +str(self.myclass[batch_class[3]]) + "  " +str(self.myclass[batch_class[4]]))
        for i in range(5):
            image = QtGui.QPixmap(self.query_image_root[self.random_indexs[i]]).scaled(self.ui.train_img1.width(),
                                                                                       self.ui.train_img1.height())
            self.pix_query_images[0][i].setPixmap(image)


    def on_calc(self):
        for i in range(5):
            self.pix_query_images[0][i].setPixmap(QPixmap(""))
            self.pix_query_images[1][i].setPixmap(QPixmap(""))
            self.pix_query_images[2][i].setPixmap(QPixmap(""))

        # calculate features
        sample_features = self.feature_encoder(Variable(self.sample_images).cuda(GPU))  # 5x64
        test_features = self.feature_encoder(Variable(self.test_images).cuda(GPU))  # 20x64

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(5, 1, 1, 1, 1)
        test_features_ext = test_features.unsqueeze(0).repeat(1 * CLASS_NUM, 1, 1, 1, 1)
        test_features_ext = torch.transpose(test_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
        relations = self.relation_network(relation_pairs).view(-1, CLASS_NUM)

        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.cpu().numpy().tolist()
        self.ui.TextEdit.appendPlainText("计算结果： " + str(predict_labels[0]) + "  " + str(predict_labels[1]) + "  " + str(
            predict_labels[2]) + "  " + str(predict_labels[3]) + "  " + str(predict_labels[4]))

        hindex = [0, 0, 0, 0, 0]
        for i in range(5):
            image = QtGui.QPixmap(self.query_image_root[self.random_indexs[i]]).scaled(self.ui.train_img1.width(),
                                                                                       self.ui.train_img1.height())
            self.pix_query_images[hindex[predict_labels[i]]][predict_labels[i]].setPixmap(image)
            hindex[predict_labels[i]] += 1
            if hindex[predict_labels[i]] == 3:
                hindex[predict_labels[i]] = 0

    def on_rechoose(self):
        train_image_root = [os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for fold in self.image_folders
                            for i
                            in np.random.randint(0, 600, 1)]
        query_image_root = [os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for fold in self.image_folders
                            for i
                            in range(600)]

        query_image_root = list(set(query_image_root) - set(train_image_root))
        # query = [query_image_root[i] for i in np.random.randint(0, 94, TEST_NUMS)]

        self.query_image_root = query_image_root
        self.train_image_root = train_image_root

        for i in range(5):
            image = QtGui.QPixmap(train_image_root[i]).scaled(self.ui.train_img1.width(), self.ui.train_img1.height())
            self.pix_train_images[i].setPixmap(image)

app = QApplication([])
stats = Show_widget()
stats.ui.show()
app.exec_()