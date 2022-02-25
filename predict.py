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


def unzip_infer_data(src_path,target_path):
    '''
    解压预测数据集
    '''
    if(not os.path.isdir(target_path)):     
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def load_image(img_path):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path) 
    if img.mode != 'RGB': 
        img = img.convert('RGB') 
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img).astype('float32') 
    img = img.transpose((2, 0, 1))  # HWC to CHW 
    img = img/255                # 像素值归一化 
    return img



if __name__ == '__main__':
    infer_src_path = '/home/aistudio/data/data129412/sports_test.zip'
    infer_dst_path = '/home/aistudio/data/sports_test'
    unzip_infer_data(infer_src_path,infer_dst_path)


    label_dic = train_parameters['label_dict']

    '''
    模型预测
    '''
    with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):   #使用GPU进行训练
    ##with fluid.dygraph.guard():                            #使用CPU进行训练 
        model = ResNet("resnet", train_parameters["network_resnet"]["layer"], train_parameters["class_dim"])
        model_dict, _ = fluid.load_dygraph("MyResNet_best")    
        #model_dict, _ = fluid.dygraph.load_dygraph("save_dir/ResNet/model_best")
        model.load_dict(model_dict)
        model.eval()

        #展示预测图片
        infer_path='data/sports_test/field_hockey' 
        infer_imag=os.path.join(infer_path,'4.jpg')        
        img = Image.open(infer_imag)
        plt.imshow(img)          #根据数组绘制图像
        plt.show()               #显示图像

        #对预测图片进行预处理
        infer_imgs = []
        infer_path='data/sports_test/field_hockey'
        img_list = os.listdir(infer_path)
        if '__MACOSX' in img_list:
            img_list.remove('__MACOSX')    
        for imgfn in img_list:
            infer_imag=os.path.join(infer_path,imgfn)
            infer_imgs.append(load_image(infer_imag))
        infer_imgs = np.array(infer_imgs)
        
        for  i in range(len(infer_imgs)):
            data = infer_imgs[i]
            dy_x_data = np.array(data).astype('float32')
            dy_x_data=dy_x_data[np.newaxis,:, : ,:]
            img = fluid.dygraph.to_variable(dy_x_data)
            out = model(img)
            lab = np.argmax(out.numpy())  #argmax():返回最大数的索引
            print("第{}个样本,被预测为：{},真实标签为：{}".format(i+1,label_dic[str(lab)],infer_path.split('/')[-1].split(".")[0]))  
    print("结束")