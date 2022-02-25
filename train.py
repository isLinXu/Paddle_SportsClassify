
#导入需要的包
import os
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import matplotlib.pyplot as plt


'''
参数配置
'''
batch_size = 32  ##
num_epochs = 40 ##后面有改动
train_parameters = {
    "input_size": [3, 224, 224],                              #输入图片的shape
    "class_dim": 100,                                          #分类数
    "src_path":"/home/aistudio/data/data129412/sports_trains.zip",            #原始数据集路径
    "target_path":"/home/aistudio/data/dataset",              #要解压的路径 
    "train_list_path": "./train_data.txt",                    #train_data.txt路径
    "eval_list_path": "./val_data.txt",                       #eval_data.txt路径
    "readme_path": "/home/aistudio/data/readme.json",         #readme.json路径
    "label_dict":{},                                          #标签字典
    "image_count": -1,                                        # 训练图片数量
    "train_batch_size": batch_size,                           #训练时每个批次的大小
    "num_epochs":  num_epochs,                                #训练轮数
    "mode": "train",                                          #工作模式
                                            
    "network_resnet": {                 #ResNet
        "layer": 50                     #ResNet的层数
    },                                              
    "continue_train": False,            # 是否接着上一次保存的参数接着训练
    "regenerat_imgs": False,            # 是否生成增强图像文件，True强制重新生成，慢
    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值
    "use_gpu": True,
    "use_image_enhance": False,
    "image_enhance_strategy": {  # 图像增强相关策略
        "need_distort": False,    # 是否启用图像颜色增强
        "need_rotate": False,     # 是否需要增加随机角度
        "need_crop": False,       # 是否要增加裁剪
        "need_flip": False,       # 是否要增加水平随机翻转
        "need_expand": False,     # 是否要增加扩展
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },
    "learning_strategy": {                                    #优化函数相关的配置
        #"lr": 0.00125,                                        #超参数学习率        
        "lr": 0.000125,                                        #超参数学习率        
        "name": "cosine_decay",
        "batch_size": batch_size,
        "epochs": [40, 80, 100],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    },  
    "early_stop": {  
        "sample_frequency": 50,  
        "successive_limit": 3,  
        "good_acc1": 0.9975 #0.92  
    },  
    "rms_strategy": {  
        "learning_rate": 0.001,
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "momentum_strategy": {  
        #"learning_rate": 0.001,
        "learning_rate": 0.0001,    
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "sgd_strategy": {  
        "learning_rate": 0.001,
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "adam_strategy": {  
        "learning_rate": 0.002          
    }, 
    "adamax_strategy": {  
        "learning_rate": 0.00125  
    }      
}


if __name__ == '__main__':
    # DATA_PATH = "/home/linxu/Downloads/GoogleDownload/CaptchaDataset-master/CaptchaDataset-master/Classify_Dataset"
    # Reader(DATA_PATH).print_sample(1)
    print('start train!')

