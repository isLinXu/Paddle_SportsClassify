
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
解压原始数据到指定路径
'''
unzip_data(src_path,target_path)

#每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
with open(eval_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
    
#生成数据列表   
get_data_list(target_path,train_list_path,eval_list_path)

'''
构造数据提供器
'''
#训练集和验证集调用同样的函数，但是工作模式这个参数不一样。
train_reader = paddle.batch(custom_reader(train_list_path, 'train'),
                            batch_size=batch_size,
                            drop_last=True)
eval_reader = paddle.batch(custom_reader(eval_list_path, 'eval'),
                            batch_size=batch_size,
                            drop_last=True)


if __name__ == '__main__':

    
    '''
    参数初始化
    '''
    src_path=train_parameters['src_path']
    target_path=train_parameters['target_path']
    train_list_path=train_parameters['train_list_path']
    eval_list_path=train_parameters['eval_list_path']
    batch_size=train_parameters['train_batch_size']
