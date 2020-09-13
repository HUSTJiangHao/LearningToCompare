import PySide2
from PySide2.QtWidgets import QApplication,QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
from  PySide2.QtCore import QFile
from PySide2 import QtGui
import os
import numpy as np

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
        qfile_stats = QFile('omniglot_one_shot.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        self.ui = QUiLoader().load(qfile_stats)
        self.image_folders = []
        self.encode_path = ''
        self.relation_path = ''

        self.ui.button_select_folds.clicked.connect(self.on_select_folders)
        self.ui.button_load_model.clicked.connect(self.on_load_model)

    def on_select_folders(self):

        dialog = QFileDialog(directory="../../datas/omniglot_resized")
        dialog.setFileMode(QFileDialog.Directory)
        for i in range(5):
            if dialog.exec_():
                fileName = dialog.selectedFiles()
                self.image_folders.append(fileName[0])
                self.ui.TextEdit.appendPlainText(" class："+ str(i)+ "  "+ fileName[0].split("/")[-2] + '/' + fileName[0].split("/")[-1])

        train_image_root = [os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for fold in self.image_folders for i
                            in np.random.randint(0, 19, 1)]
        query_image_root = [os.path.join(fold, os.listdir(fold)[i]).replace("\\", "/") for fold in self.image_folders for i
                            in range(20)]

        query_image_root = list(set(query_image_root) - set(train_image_root))

        query = [query_image_root[i] for i in np.random.randint(0, 94, TEST_NUMS)]
        query_image_root = query

        image1 = QtGui.QPixmap(train_image_root[0]).scaled(self.ui.train_img1.width(), self.ui.train_img1.height())
        self.ui.train_img1.setPixmap(image1)
        image2 = QtGui.QPixmap(train_image_root[1]).scaled(self.ui.train_img1.width(), self.ui.train_img1.height())
        self.ui.train_img2.setPixmap(image2)
        image3 = QtGui.QPixmap(train_image_root[2]).scaled(self.ui.train_img1.width(), self.ui.train_img1.height())
        self.ui.train_img3.setPixmap(image3)
        image4 = QtGui.QPixmap(train_image_root[3]).scaled(self.ui.train_img1.width(), self.ui.train_img1.height())
        self.ui.train_img4.setPixmap(image4)
        image5 = QtGui.QPixmap(train_image_root[4]).scaled(self.ui.train_img1.width(), self.ui.train_img1.height())
        self.ui.train_img5.setPixmap(image5)


    def on_load_model(self):
        dialog = QFileDialog(caption = "选择编码网络",directory="../models")
        if dialog.exec_():
            fileName = dialog.selectedFiles()
            self.encode_path = fileName[0]
            self.ui.TextEdit.appendPlainText("编码网络模型：" +  fileName[0])

        dialog = QFileDialog(caption="选择关系网络" ,directory = "../models")
        if dialog.exec_():
            fileName = dialog.selectedFiles()
            self.relation_path = fileName[0]
            self.ui.TextEdit.appendPlainText("关系网络模型：" +  fileName[0])



app = QApplication([])
stats = Show_widget()
stats.ui.show()
app.exec_()