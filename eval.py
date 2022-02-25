
#导入需要的包
from json.tool import main
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

#定义eval_net函数
# print(train_parameters)

def eval_net(reader, model):
    acc_set = []
    
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int')
        y_data = y_data[:, np.newaxis]
        img = fluid.dygraph.to_variable(dy_x_data)
        label = fluid.dygraph.to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)

        #        out, acc = model(img, label)
        #        lab = np.argsort(out.numpy())
        #        accs.append(acc.numpy()[0]) 

        acc_set.append(float(acc.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()

    return acc_val_mean



def eval_model():
    '''
    模型评估
    '''
    import numpy as np
    import argparse
    import ast
    import paddle
    import paddle.fluid as fluid
    from paddle.fluid.layer_helper import LayerHelper
    from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
    from paddle.fluid.dygraph.base import to_variable
    from paddle.fluid import framework
    import math
    import sys
    from paddle.fluid.param_attr import ParamAttr

    # resnet层数定义，要改一下
    #train_parameters["layer"] = 50

    with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):   #使用GPU进行训练
    ##with fluid.dygraph.guard():                            #使用CPU进行训练    
        model = ResNet("resnet", train_parameters["network_resnet"]["layer"], train_parameters["class_dim"])
        model_dict, _ = fluid.load_dygraph("MyResNet_best")    
        #model_dict, _ = fluid.dygraph.load_dygraph("save_dir/ResNet/model_best")
        model.load_dict(model_dict)
        model.eval()

        accs = []
        for batch_id, data in enumerate(eval_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32').reshape(-1,3,224,224)
            y_data = np.array([x[1] for x in data]).astype('int').reshape(-1,1)
            
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)

            out, acc = model(img, label)
            lab = np.argsort(out.numpy())
            accs.append(acc.numpy()[0])

        avg_acc = np.mean(accs)
        #print(np.mean(accs))
        print("模型校验avg_acc=",avg_acc)

