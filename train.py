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

if __name__ == '__main__':
    # DATA_PATH = "/home/linxu/Downloads/GoogleDownload/CaptchaDataset-master/CaptchaDataset-master/Classify_Dataset"
    # Reader(DATA_PATH).print_sample(1)
    print('start train!')
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

'''
模型训练，继续炼丹
'''
train_parameters["num_epochs"] = 40  # 设置轮次
epochs_num = train_parameters["num_epochs"]
batch_size = train_parameters["train_batch_size"] # train_parameters["learning_strategy"]["batch_size"]
total_images=train_parameters["image_count"]
stepsnumb = int(math.ceil(float(total_images) / batch_size))        

# resnet层数定义，要改一下
#train_parameters["layer"] = 50  

with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):   #使用GPU进行训练
##with fluid.dygraph.guard():                            #使用CPU进行训练
    print('class_dims:',train_parameters['class_dim'])
    print('label_dict:',train_parameters['label_dict'])

    best_acc = 0    
    best_epc = -1 
    eval_epchnumber = 0
    all_eval_avgacc = []
    all_eval_iters = []

    all_train_iter =0
    all_train_iters=[]
    all_train_costs=[]
    all_train_accs=[]

    model = ResNet("resnet", train_parameters["network_resnet"]["layer"], train_parameters["class_dim"])

    #"""
    if True:        
        try:
            if os.path.exists('MyResNet_best.pdparams'):
                print('try model file MyResNet_best. Loading...')
                model_dict, _ = fluid.load_dygraph('MyResNet_best')        
                if os.path.exists('beast_acc_ResNet.txt'):
                    with open('beast_acc_ResNet.txt', "r") as f:
                        best_acc = float(f.readline())
            else:
                print('try model file MyResNet. Loading...')
                model_dict, _ = fluid.load_dygraph('MyResNet')        
                if os.path.exists('acc_ResNet.txt'):
                    with open('acc_ResNet.txt', "r") as f:
                        best_acc = float(f.readline())
            #防止上一次acc太大，导致本次训练结果不存储了
            start_acc = min(0.92,train_parameters["early_stop"]["good_acc1"])
            if best_acc>=start_acc:
                best_acc=start_acc        
            model.load_dict(model_dict) #加载模型参数  
        except Exception as e:
            print(e)                
        print('model initialization finished.')
    #"""   

    #后面代码会切换工作模式
    model.train() #训练模式

    paramsList=model.parameters()
    params = train_parameters
    total_images = params["image_count"]
    ls = params["learning_strategy"]
    batch_size = ls["batch_size"]
    step = int(math.ceil(float(total_images) / batch_size))
    bd = [step * e for e in ls["epochs"]]
    # 固定学习率
    lr = 0.0000625 #params["learning_strategy"]["lr"]  #0.00125
    num_epochs = params["num_epochs"]
    regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1)
    learning_rate=lr

    # 学习率衰减
    # learning_rate=fluid.layers.cosine_decay(
    #    learning_rate=lr, step_each_epoch=step, epochs=num_epochs)
    #momentum_rate = 0.9

    #定义优化方法 optimizer_momentum_setting, optimizer_sgd_setting, optimizer_rms_setting, optimizer_adam_setting, optimizer_Adamax_setting
    #optimizer = optimizer_momentum_setting(model.parameters())
    #optimizer = fluid.optimizer.Momentum(learning_rate=learning_rate,momentum=momentum_rate,regularization=regularization,parameter_list=paramsList)
    #optimizer=fluid.optimizer.SGDOptimizer(learning_rate=learning_rate, regularization=regularization, parameter_list=paramsList)
    optimizer=fluid.optimizer.AdamaxOptimizer(learning_rate=learning_rate, regularization=regularization, parameter_list=paramsList)
    #optimizer=fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, regularization=regularization, parameter_list=paramsList) 

    #epochs_num = 1
     #开始训练
    for epoch_num in range(epochs_num):
        model.train() #训练模式
         #从train_reader中获取每个批次的数据
        for batch_id, data in enumerate(train_reader()):
            #dy_x_data = np.array([x[0] for x in data]).astype('float32')
            #y_data = np.array([x[1] for x in data]).astype('int')
            dy_x_data = np.array([x[0] for x in data]).astype('float32').reshape(-1, 3,224,224)
            y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1,1)                               

            #将Numpy转换为DyGraph接收的输入
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)

            out,acc = model(img,label)
            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)

            #使用backward()方法可以执行反向网络
            avg_loss.backward()
            optimizer.minimize(avg_loss)             
            #将参数梯度清零以保证下一轮训练的正确性
            model.clear_gradients()            

            all_train_iter=all_train_iter+train_parameters['train_batch_size']
            all_train_iters.append(all_train_iter)
            all_train_costs.append(loss.numpy()[0])
            all_train_accs.append(acc.numpy()[0])
            
            dy_param_value = {}
            for param in model.parameters():
                dy_param_value[param.name] = param.numpy
                
            if batch_id % 100 == 0 or batch_id == stepsnumb-1:
                print("epoch %3d step %4d: loss: %f, acc: %f" % (epoch_num, batch_id, avg_loss.numpy(), acc.numpy()))

        if epoch_num % 1 == 0 or epoch_num == epochs_num-1:
            model.eval()      
            epoch_acc = eval_net(eval_reader, model) 
            print('  train_pass:%d,eval_acc=%f' % (epoch_num,epoch_acc))  
            eval_epchnumber = epoch_num
            all_eval_avgacc.append(epoch_acc)
            all_eval_iters.append([eval_epchnumber, epoch_acc])
                
            if best_acc < epoch_acc:  
                best_epc=epoch_num                                      
                best_acc=epoch_acc
                #保存模型参数，对应当前最好的评估结果
                fluid.save_dygraph(model.state_dict(),'MyResNet_best')
                print('    current best_eval_acc=%f in No.%d epoch' % (best_acc,best_epc)) 
                print('    MyResNet_best模型已保存')
                with open('beast_acc_ResNet.txt', "w") as f:
                    f.write(str(best_acc))
                #fluid.dygraph.save_dygraph(model.state_dict(), "save_dir/ResNet/model_best")
                #fluid.dygraph.save_dygraph(optimizer.state_dict(), "save_dir/ResNet/model_best")    

            #训练过程结果显示
            draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning loss","trainning acc")      
    
    #保存模型参数，但不一定是最好评估结果对应的模型
    fluid.save_dygraph(model.state_dict(), "MyResNet")   
    print('MyResNetG模型已保存')
    print("Final loss: {}".format(avg_loss.numpy()))
    #fluid.dygraph.save_dygraph(model.state_dict(), "save_dir/model")
    #fluid.dygraph.save_dygraph(optimizer.state_dict(), "save_dir/model")   
    with open('acc_ResNet.txt', "w") as f:
        f.write(str(epoch_acc)) 

    draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning loss","trainning acc")  
    draw_process("trainning loss","red",all_train_iters,all_train_costs,"trainning loss")
    draw_process("trainning acc","green",all_train_iters,all_train_accs,"trainning acc")          

