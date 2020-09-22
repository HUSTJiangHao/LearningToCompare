import cv2
import sys
import os

origin_root_path = 'D:/_CODE/LearningToCompare/datas/miniImagenet/test'
target_root_path = 'D:\_CODE\LearningToCompare\datas\miniImagenet\_PNG'

test_class = os.listdir(origin_root_path)
origin_class_folds_path =  [os.path.join(origin_root_path, i)  for i in os.listdir(origin_root_path)]

target_class_folds_path = [os.path.join(target_root_path, i)  for i in os.listdir(origin_root_path)]

# for fold in target_class_folds_path:
#     os.mkdir(fold)

for i in range(len(origin_class_folds_path)):
    origin_fold = origin_class_folds_path[i]
    target_fold = target_class_folds_path[i]

    for img in os.listdir(origin_fold):
        img_path = os.path.join(origin_fold, img)
        img_name = img.split('.')[-2]
        img_tar_path = os.path.join(target_fold, img_name) + '.png'

        print(img_tar_path)
        src = cv2.imread(img_path)
        cv2.imwrite(img_tar_path, src)

