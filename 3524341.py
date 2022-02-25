#!/usr/bin/env python
# coding: utf-8

# # 基于PaddleClas的鸟类图片分类

# ## 一、项目背景介绍、
# * 本人对无穷小亮的科普视频很感兴趣，博主通过自身知识对各类动植物进行讲解的形式让人耳目一新。在各类动植物中，我比较喜欢鸟类，观看学习之余，我想到可以通过飞桨平台的工具实现对不同鸟类的图片进行信息识别。这样一个方便使用的小项目，既可以简化信息搜索的难度，还可以助力知识的普及。
# 

# ## 二、数据介绍
# 
# * （[275 种鸟类](https://aistudio.baidu.com/aistudio/datasetdetail/99214/0)）数据集
# 
# * 275 种鸟类的数据集。39364张训练图像，1375张测试图像（每个物种5张）和1375张验证图像（每个物种5张。所有图像均为jpg格式的224 X 224 X 3彩色图像。数据集包括训练集、测试集和验证集。每组包含 275 个子目录，每个鸟种一个。

# （1）数据处理：准备数据以及安装环境依赖

# In[1]:


get_ipython().system('unzip -oq /home/aistudio/data/data99214/Bird_Dataset.zip -d work/')


# In[4]:


get_ipython().run_line_magic('cd', 'work/')
get_ipython().system('git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.3')
get_ipython().run_line_magic('cd', 'PaddleClas/')
get_ipython().system('pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple')
get_ipython().system('export PYTHONPATH=path_to_PaddleClas:$PYTHONPATH')


# In[3]:


get_ipython().system('pip install daal --upgrade --ignore-installed daal')


# 训练文件：work/PaddleClas/tools/train.py
# 
# 训练配置文件：work/PaddleClas/ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml

# （2）样本的可视化展示

# In[1]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image_path_list = ['work/birds/valid/AFRICAN FIREFINCH/1.jpg']
plt.imshow(cv2.imread('work/birds/valid/AFRICAN FIREFINCH/1.jpg'))


# （3）定义数据集

# ```
# # 读取label配置
# %cd ~
# import os
# import random
# import cv2
# import numpy as np
# import shutil
# #import json
# 
# with open("work/birds/birds.csv",'r') as load_f:
#     label_map = json.load(load_f)
#     print(label_map)
# 
# ```

# In[3]:


import paddle
import numpy as np
import paddle.vision.transforms as T


class MyImageNetDataset(paddle.io.Dataset):
    def __init__(self,num_samples):
        super(MyImageNetDataset, self).__init__()

        self.num_samples = num_samples
        self.transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=127.5, std=127.5)])

    def __getitem__(self, index):
        image = np.random.randint(low=0, high=256, size=(512, 512, 3))
        

        image = image.astype('float32')
        

        image = self.transform(image)

        return image

    def __len__(self):
        return self.num_samples


# In[4]:


import numpy as np
from PIL import Image
from paddle.vision.transforms import ColorJitter


transform = ColorJitter(0.4, 0.4, 0.4, 0.4)

for (root , dirs, files ) in os.walk("work/birds", topdown=False):
    for name in files:
        abs_name=os.path.join(root, name)
        print(abs_name)
        img = Image.open(abs_name)
        rotated_img = transform(img)
        print(rotated_img.size)
        fake_name=os.path.join(root,"ColorJitter_"+ name)
        print(fake_name)
        rotated_img.save(fake_name)


# （4）聚合数据

# In[21]:


data_path = "/home/work/birds"
train_folder = "train"
test_folder = "test"
data_list = []
for index in range(20):
    print(index,label_map[str(index)])
    sub_folder=label_map[str(index)]
    abs_sub_folder=os.path.join(data_path, train_folder,sub_folder)
    sub_list=os.listdir(abs_sub_folder)
    for sub_name in sub_list:
        data_list.append(os.path.join(train_folder, sub_folder, sub_name) + ' ' + str(index) + '\n')
    print("Finished train_folder: {}".format(sub_folder))
print("Finished data_list length: {}".format(len(data_list)))

val_ratio = 0.1

# save file
def save_file(list, txt):
    myfile=os.path.join('dataset',txt)
    if os.path.exists(myfile):
        os.remove(myfile)
    with open(myfile, "a") as f:
        f.writelines(list)


random.shuffle(data_list)
val_number = int(val_ratio * len(data_list))
train_list = data_list[val_number:]
val_list = data_list[0:val_number]
print("full data size: {}   train_list size: {}   val_list size: {}".format(len(data_list), len(data_list)-val_number,val_number))

save_file(train_list,  'train_list.txt')
save_file(val_list,  'test_list.txt')


# ## 三、模型介绍
# * PaddleCV飞桨视觉模型库，提供大量高精度、高推理速度、经过产业充分验证的智能视觉模型，覆盖各类任务场景。PaddleClas、PaddleDet和PaddleSeg等端到端的开发套件，打通模型开发、训练、压缩、部署全流程，并支持超大规模分类等进阶功能，为开发者提供高效顺畅的开发体验。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/25542963e5284cf0a6be6f6fd3ca4845e1c3d578326c4a85a1338846bd51944a)
# 

# ## 四、模型训练

# （1）开始训练

# In[5]:


get_ipython().run_line_magic('cd', '/home/aistudio/work/PaddleClas/')


# ```
# # global configs
# Global:
#   checkpoints: null
#   pretrained_model: null
#   output_dir: ./output/
#   device: gpu
#   class_num: 275
#   save_interval: 10
#   eval_during_train: True
#   eval_interval: 10
#   epochs: 250
#   print_batch_step: 10
#   use_visualdl: False
#   # used for static mode and model export
#   image_shape: [3, 224, 224]
#   # save_inference_dir: ./inference
# 
# # model architecture
# Arch:
#   name: ResNet50_vd
#  
# # loss function config for traing/eval process
# Loss:
#   Train:
#     - CELoss:
#         weight: 1.0
#   Eval:
#     - CELoss:
#         weight: 1.0
# 
# 
# Optimizer:
#   name: Momentum
#   momentum: 0.9
#   lr:
#     name: Cosine
#     learning_rate: 0.0125
#     warmup_epoch: 5
#   regularizer:
#     name: 'L2'
#     coeff: 0.00001
# 
# 
# # data loader for train and eval
# DataLoader:
#   Train:
#     dataset:
#       name: ImageNetDataset
#       image_root: /home/aistudio/dataset/
#       cls_label_path: /home/aistudio/dataset/train_list.txt
#       transform_ops:
#         - DecodeImage:
#             to_rgb: True
#             channel_first: False
#         - RandCropImage:
#             size: 224
#         - RandFlipImage:
#             flip_code: 1
#         - RandAugment:
#             num_layers: 2
#             magnitude: 5
#         - NormalizeImage:
#             scale: 1.0/255.0
#             mean: [0.485, 0.456, 0.406]
#             std: [0.229, 0.224, 0.225]
#             order: ''
# 
#     sampler:
#       name: DistributedBatchSampler
#       batch_size: 300
#       drop_last: False
#       shuffle: True
#     loader:
#       num_workers: 4
#       use_shared_memory: True
# 
#   Eval:
#     dataset: 
#       name: ImageNetDataset
#       image_root: /home/work/birds
#       cls_label_path: /home/work/birds/test.txt
#       transform_ops:
#         - DecodeImage:
#             to_rgb: True
#             channel_first: False
#         - ResizeImage:
#             resize_short: 256
#         - CropImage:
#             size: 224
#         - NormalizeImage:
#             scale: 1.0/255.0
#             mean: [0.485, 0.456, 0.406]
#             std: [0.229, 0.224, 0.225]
#             order: ''
#     sampler:
#       name: DistributedBatchSampler
#       batch_size: 128
#       drop_last: False
#       shuffle: False
#     loader:
#       num_workers: 4
#       use_shared_memory: True
# 
# Metric:
#   Train:
#     - TopkAcc:
#         topk: [1, 5]
#   Eval:
#     - TopkAcc:
#         topk: [1, 5]
# ```

# （2）模型选取

# In[7]:


get_ipython().system('python  tools/train.py  -c  ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml')


# （7）验证

# In[9]:


get_ipython().system('python -m paddle.distributed.launch     --gpus="0"     tools/eval.py     -o ARCHITECTURE.name="EfficientNetB0"     -o pretrained_model=\'./output/EfficientNetB0/best_model_train/ppcls\'')


# （3）导出

# In[10]:


get_ipython().system("python tools/export_model.py     --model='EfficientNetB0'     --pretrained_model='./output_finetune/EfficientNetB0/best_model_finetune/ppcls'     --output_path='./output_finetune/EfficientNetB0_infer'")


# （4）预测

# ```
# !python tools/infer/predict.py \
#     -m=./output_finetune/EfficientNetB0_infer/model \
#     -p=./output_finetune/EfficientNetB0_infer/params \
#     -i=/home/aistudio/dataset/Butterfly20_test \
#     # -i=./dataset/sgt/test/jpg \
#     --use_gpu=1
# ```

# ```
# # 读取,并转换为value，
# %cd /home/aistudio
# import fileinput
# 
# value_list = []
# for line in fileinput.input("/home/aistudio/result_id.txt"):
#     value_list.append(label_map[line.strip('\n')])
# print(len(value_list))
# 
# 
# ```

# (5)结果处理

# ```
# # 保存
# savename = '/home/aistudio/result.txt'
# with open(savename, 'w', newline='') as f:
#     for line in value_list:
#         f.write(line+"\n")
# ```

# ## 五、总结
# 通过系列课程让本人对整体的项目流程有了一定的了解，但因本人基础知识欠缺，对初见的各种变量名和指令还待了解，对异常报错的经验还比较少，仅能做到通过参考相关文档模拟项目流程，故本人还应以后持续学习，进一步完善本项目及自己的深度学习知识体系。
# 

# 
# 参考资料：
#        1.[30分钟玩转PaddleClas](https://paddleclas.readthedocs.io/zh_CN/latest/tutorials/quick_start.html)    2.[用PaddleClas解决蝴蝶分类](https://aistudio.baidu.com/aistudio/projectdetail/3525369?forkThirdPart=1)
