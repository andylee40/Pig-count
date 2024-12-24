#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# file_path='/home/u7412932/multiple_car/yolov7/pig_heatstress_2024_0223_2-2/valid'
# dist_path='/home/u7412932/multiple_car/yolov7/pig_heatstress_2024_shuffle/train'
# for key in ['labels','images']:
#     for file in os.listdir(file_path+'/'+key):
#         shutil.copy2(file_path+'/'+key+'/'+file, dist_path+'/'+key+'/'+file)


# len(os.listdir('/home/u7412932/multiple_car/yolov7/pig_heatstress_2024_shuffle/train/images'))


#原始資料路徑
data_path = '/home/u7412932/pigcount_0819/yolov7/pig_heatstress_2024_shuffle/train'
data_type = ['images', 'labels']
image_list = [x.resolve() for x in Path(data_path).glob('images/*.jpg')]
print(len(image_list))

train_type = ['train', 'test', 'valid']
#存放資料夾
save_path = '/home/u7412932/pigcount_0819/yolov7/yolov7_multi_2024_shuffle_0926'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in train_type:
  for j in data_type:
    os.makedirs(os.path.join(save_path, i, j))


# 先將資料集分成 80% train+valid 和 20% test
train_valid_list, test_list = train_test_split(image_list, test_size=0.2, random_state=1)
# 再將 train+valid 資料集分成 75% train 和 25% valid，這樣最終 train 佔整體的 60%，valid 佔 20%
train_list, valid_list = train_test_split(train_valid_list, test_size=0.25, random_state=1)
print(len(train_list), len(valid_list), len(test_list))


dir_dic = {
    'train':train_list,
    'test':test_list,
    'valid':valid_list
}



for name in tqdm(list(dir_dic.keys())):
  paths = dir_dic[name]
  for path in paths:
    # image
    file_name = path.name
    shutil.copy2(str(path), f'/home/u7412932/pigcount_0819/yolov7/yolov7_multi_2024_shuffle_0926/{name}/images')
    # label
    label_path = str(path).replace('images', 'labels').replace('.jpg', '.txt')
    shutil.copy2(label_path, f'/home/u7412932/pigcount_0819/yolov7/yolov7_multi_2024_shuffle_0926/{name}/labels')
    with open(f'/home/u7412932/pigcount_0819/yolov7/yolov7_multi_2024_shuffle_0926/{name}.txt', 'a') as file:
      file.write(str(path))
      file.write('\n')


# In[9]:


# get_ipython().system('tar zcvf pig20240926.tar.gz /home/u7412932/pigcount_0819/yolov7/yolov7_multi_2024_shuffle_0926')


# In[ ]:




