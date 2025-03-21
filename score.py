import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
from PIL import Image
from tqdm import tqdm



# 文件夹路径
gt_folder = 'Dataset/PH2/test/masks'
pred_folder = 'Predictions/DUN/PH2'

# 获取文件列表
gt_files = os.listdir(gt_folder)
pred_files = os.listdir(pred_folder)

# 初始化列表用于存储所有的预测结果和真实标签
pred_vector_list = []
gt_vector_list = []

with tqdm(total=len(pred_files)) as t:
    for gt_file, pred_file in zip(gt_files, pred_files):
        if gt_file.endswith('.png') and pred_file.endswith('.png'):

            # Pred图像
            pred_img_path = os.path.join(pred_folder, pred_file)
            image = Image.open(pred_img_path)
            image = image.resize((256, 256))
            # 将图像转换为灰度图像
            image = image.convert('L')
            # 将图像转换为NumPy数组
            image_array = np.array(image)
            # 设置阈值
            threshold = 0.5
            # 将图像二值化
            binary_image = (image_array > threshold).astype(int)
            # 将二值化的图像向量化
            pred_vector = binary_image.flatten()
            pred_vector_list.append(pred_vector)

            # GT图像
            gt_file_name = gt_file.split('.')[0] + '.png'  # 假设真实标签文件名为"图像文件名_gt.png"

            gt_image_path = os.path.join(gt_folder, gt_file_name)
            gt_image = Image.open(gt_image_path)
            gt_image = gt_image.resize((256, 256))
            gt_image = gt_image.convert('L')
            gt_array = np.array(gt_image)
            gt_binary = (gt_array > threshold).astype(int)
            gt_vector = gt_binary.flatten()
            gt_vector_list.append(gt_vector)
        t.update(1)



pred_matrix = np.array(pred_vector_list)
gt_matrix = np.array(gt_vector_list)

# 计算F1值
f1 = f1_score(gt_matrix, pred_matrix, average='micro')

# 计算mIoU（平均交并比）
miou = jaccard_score(gt_matrix, pred_matrix, average='micro')

# 计算召回率
recall = recall_score(gt_matrix, pred_matrix, average='micro')

# 计算精确度
precision = precision_score(gt_matrix, pred_matrix, average='micro')

print('F1 Score: {:.4f}'.format(f1))
print('mIoU: {:.4f}'.format(miou))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
