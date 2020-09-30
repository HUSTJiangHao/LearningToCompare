import PySide2
from PySide2.QtWidgets import QApplication,QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from  PySide2.QtCore import QFile
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
#import task_generator as tg
import os
import math
import miniimgnet_few_shot_model as this_model

# Hyper Parameters
FEATURE_DIM = 64
RELATION_DIM = 8
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 5
BATCH_NUM_PER_CLASS = 10
EPISODE = 10
TEST_EPISODE = 600
LEARNING_RATE = 0.001
GPU = 0
HIDDEN_UNIT = 10

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class Show_widget:

    def __init__(self):
        qfile_stats = QFile('miniimgnet_few_shot.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        self.ui = QUiLoader().load(qfile_stats)

        self.image_folders = []
        self.encode_path = ''
        self.relation_path = ''
        self.query_image_root = []  # 查询集路径
        self.train_image_root = []
        self.feature_encoder = this_model.CNNEncoder()
        self.relation_network = this_model.RelationNetwork(FEATURE_DIM, RELATION_DIM)

        self.ui.button_select_folds.clicked.connect(self.on_select_folders)
        self.ui.button_load_model.clicked.connect(self.on_load_model)
        self.ui.button_multi_show.clicked.connect(self.on_batch_show)

        self.pix_train_images = [[self.ui.train_11 ,self.ui.train_12 ,self.ui.train_13 ,self.ui.train_14 ,self.ui.train_15],
                                 [self.ui.train_21, self.ui.train_22, self.ui.train_23, self.ui.train_24, self.ui.train_25],
                                 [self.ui.train_31, self.ui.train_32, self.ui.train_33, self.ui.train_34, self.ui.train_35],
                                 [self.ui.train_41, self.ui.train_42, self.ui.train_43, self.ui.train_44, self.ui.train_45],
                                 [self.ui.train_51, self.ui.train_52, self.ui.train_53, self.ui.train_54, self.ui.train_55]]

        self.pix_query_images = [self.ui.test_1, self.ui.test_2, self.ui.test_3, self.ui.test_4,
                                 self.ui.test_5]

    def on_select_folders(self):

        dialog = QFileDialog(directory="../../datas/miniimagenet/_PNG")
        dialog.setFileMode(QFileDialog.Directory)

        self.image_folders = []
        for i in range(5):
            if dialog.exec_():
                fileName = dialog.selectedFiles()
                self.image_folders.append(fileName[0])
                self.ui.TextEdit.appendPlainText(
                    " class：" + str(i) + "  " + fileName[0].split("/")[-2] + '/' + fileName[0].split("/")[-1])

        train_image_root = [[os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for i in np.random.randint(0, 19, 5)]
                             for fold in self.image_folders]
        query_image_root = [[os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for i in range(20)]
                            for fold in self.image_folders]


        query_image_root = [list(set(query_image_root[i]) - set(train_image_root[i])) for i in range(5)]
        query = query_image_root[0] + query_image_root[1] + query_image_root[2] + query_image_root[3] + query_image_root[4]

        self.query_image_root = query
        self.train_image_root = train_image_root

        for i in range(5):
            for j in range(5):
                image = QtGui.QPixmap(train_image_root[i][j]).scaled(self.ui.train_11.width(), self.ui.train_11.height())
                self.pix_train_images[i][j].setPixmap(image)

    def on_load_model(self):
        dialog = QFileDialog(caption="选择编码网络", directory="../models")
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

        sample_images = torch.zeros(25, 3, 84, 84)
        test_images = torch.zeros(5, 3, 84, 84)
        for i in range(5):
            for j in range(5):
                image = Image.open(self.train_image_root[i][j])
                image = image.convert('RGB')

                normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
                transform = transforms.Compose([transforms.ToTensor(), normalize])
                image = transform(image)

                sample_images[i*5 + j] = image


        random_indexs = np.random.randint(0, len(self.query_image_root) - 1, 5)
        for i in range(5):
            image = Image.open(self.query_image_root[random_indexs[i]])
            image = image.convert('RGB')  # 灰度图像

            normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            image = transform(image)

            test_images[i] = image

        for i in range(5):
            image = QtGui.QPixmap(self.query_image_root[random_indexs[i]]).scaled(self.ui.test_1.width(),
                                                                                  self.ui.test_1.height())
            self.pix_query_images[i].setPixmap(image)

        sample_features = self.feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
        test_features = self.feature_encoder(Variable(test_images).cuda(GPU))  # 20x64

        # calculate relations
        sample_features = self.feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        test_features = self.feature_encoder(Variable(test_images).cuda(GPU))  # 20x64

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
        self.ui.label_testclass.setText(
            "class：  " + str(predict_labels[0]) + "                 " + str(predict_labels[1]) +
            "                 " + str(predict_labels[2]) + "                 " + str(predict_labels[3]) +
            "                 " + str(predict_labels[4]))


app = QApplication([])
stats = Show_widget()
stats.ui.show()
app.exec_()