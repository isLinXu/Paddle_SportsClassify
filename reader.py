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


# reader
def custom_reader(file_list,mode):
    '''
    自定义data_reader
    '''
    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            ## 打乱次序
            random.shuffle(lines)
            for line in lines:
                if mode == 'train' or mode == 'eval':
                    img_path, lab = line.strip().split('\t')
                    img = Image.open(img_path) 
                    if img.mode != 'RGB': 
                        img = img.convert('RGB') 
                    #使用在线增强方式，不用的话，把下一段代码注释掉
                    #验证模式下不用增强，所以加了一个工作模式条件判断
                    #"""                    
                    if mode == 'train': 
                        #在线增强方式：
                        if (train_parameters["use_image_enhance"]):
                            img = preprocess(img, mode)  #只有在'train'模式下才执行图像增强
                    #"""
                    #图像缩放到指定大小，VGG是3x224x224
                    img = img.resize((224, 224), Image.ANTIALIAS)  ##BILINEAR
                    img = np.array(img).astype('float32') 
                    #图像数据按照所需要的格式重新排列
                    img = img.transpose((2, 0, 1))  # HWC to CHW 
                    img = img/255.0                   # 像素值归一化 
                    yield img, int(lab) 
                elif mode == 'test':
                    img_path = line.strip()
                    img = Image.open(img_path)
                    img = img.resize((224, 224), Image.ANTIALIAS)
                    img = np.array(img).astype('float32') 
                    img = img.transpose((2, 0, 1))  # HWC to CHW 
                    img = img/255.0                   # 像素值归一化                     
                    yield img                    
    return reader