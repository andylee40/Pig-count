#!/usr/bin/env python
# coding: utf-8

# In[2]:


from collections import OrderedDict
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import os, sys
import time
import os
import operator
import argparse
import shutil
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import datetime
# import mysql.connector
import requests

from pathlib import Path
from random import randint
from numpy import random
from matplotlib import pyplot as plt
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from tqdm import tqdm

#偵測車子圖片轉換
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

weights = './best.pt'
# device = select_device('0')
device = torch.device('cuda:1')
#device = select_device('cpu')
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(640, s=stride)
half = device.type != 'cpu'
if half:
    model.half()
    
history = OrderedDict()
class BYTETrackerArgs:

    #檢測框閾值區分高分與低分框
    track_thresh: float = 0.6
    #軌跡保留幀數
    track_buffer: int = 30
    #iou match 越低越嚴格
    match_thresh: float = 0.8
    #目標長寬比的閾值
    aspect_ratio_thresh: float = 3.0
    #目標面積的閾值
    min_box_area: float = 1.0
    #不使用mot20資料進行測試
    mot20: bool = False
    
byte_tracker = BYTETracker(BYTETrackerArgs())

def count_pig_v2(model=model):
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    half = device.type != 'cpu'
    if half:
        model.half()

    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    names = model.module.names if hasattr(model, 'module') else model.names

    source = '20240228_2.mkv' ## or rtsp
    cudnn.benchmark=True
    # dataset = LoadStreams(source, img_size=imgsz, stride=stride) ## when rtsp
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    start=datetime.datetime.now()
    svefile = datetime.datetime.strftime(start,'%Y-%m-%d %H:%M:%S').replace('-','_')
    svefile = svefile.replace(':','_') 
    svefile = svefile.replace(' ','_')+'.mp4'
    save_path = './{}'.format(svefile)
    
    last_api_call_time = datetime.datetime.now()  # 初始化最後一次API調用時間
    
    vid_path, vid_writer = None, None
    p_count = 0
    start2=start.strftime('%Y-%m-%d %H:%M:%S')

    
    with torch.inference_mode():
        check_list=[]
        check_time=[]
        count_pig=0
        start_count=0
        count_line=850
        save2=[]
        a=0
        for path, img, im0s, vid_cap in tqdm(dataset):

            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.70, 0.40, agnostic=False)
            
            # # 設定結束點豬判斷方式
            # if ('條件'):
            #     end=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #     try:
            #         if count_pig>5:
            #             tosql(start2,end,count_pig)
            #     except:
            #         pass
            #     break

            for j, det in enumerate(pred):
                
                # im0s=im0s[j].copy()
                # im0s=im0s[j][200:,:].copy()
                # im0s=im0s[200:,:].copy()     
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    # start_count += int((det[:,-1]==0).sum())
                    # print('是否有車 : ',int((det[:,-1]==0).sum()))
                    # check_list.append(int((det[:,-1]==0).sum()))

                    for x1,y1,x2,y2,conf,detclass in det[det[:,-1]==2].cpu().detach().numpy():
                        # print((int(x1),int(y1)),(int(x2),int(y2)))
                        cv2.rectangle(im0s, (int(x1),int(y1)),(int(x2),int(y2)), (0, 0, 255), 5)
                        
                    for x1,y1,x2,y2,conf,detclass in det[det[:,-1]==0].cpu().detach().numpy():
                        # try:
                        #     print('車子信心 : ',conf)
                        # except:
                        #     pass
                        cv2.rectangle(im0s, (int(x1),int(y1)),(int(x2),int(y2)), (255, 0, 0), 5)
                    #tracking-------------------------------
                    #得到目標Id
                    tracks = byte_tracker.update(
                                        output_results=det[det[:,-1]==2][:,:5].cpu(),
                                        img_info=im0s.shape,
                                        img_size=im0s.shape
                                    )
                    
                    for track in tracks:
                        #目標邊框左上角和右下角座標
                        xyxy = track.tlbr
                        #目標id
                        class_id = track.track_id
                        frame_id = track.frame_id
                        his_info = np.hstack([xyxy, np.array([frame_id])])
                        if class_id in history:
                            history[class_id].append(his_info)
                        else:
                            history[class_id] = [his_info]

                    save={}
                    for ccount in history.items():
                        key=str(ccount[0])
                        length=len(ccount[1])
                        save[key]=length

                        if a:
                            try:
                                if save2[-1][str(ccount[0])]==len(ccount[1]):
                                    continue
                                else:
                                    #last value
                                    last=ccount[1][-1]
                                    try:
                                        second_last=ccount[1][-2]
                                    except:
                                        second_last=0

                                    y1=last[0]
                                    y2=last[2]

                                    try:
                                        if (second_last==0):
                                            y11=0
                                            y21=0
                                        else:
                                            y11=second_last[0]
                                            y21=second_last[2]
                                    except:
                                        y11=second_last[0]
                                        y21=second_last[2]
                                    
                                    #修正追蹤不見的問題
                                    if (len(ccount[1])==1) & ((y1+(y2-y1)/2)>count_line):
                                        count_pig-=1
                                    
                                    if ((y1+(y2-y1)/2)>count_line) & ((y11+(y21-y11)/2)<count_line):
                                        count_pig+=1
                                    elif ((y1+(y2-y1)/2)>count_line) & ((y11+(y21-y11)/2)>count_line):
                                        count_pig+=0
                                    elif ((y1+(y2-y1)/2)<count_line) & ((y11+(y21-y11)/2)>count_line):
                                        count_pig-=1
                                    elif ((y1+(y2-y1)/2)<count_line) & ((y11+(y21-y11)/2)<count_line):
                                        count_pig+=0
                            except:
                                last=ccount[1][-1]
                                try:
                                    second_last=ccount[1][-2]
                                except:
                                    second_last=0

                                y1=last[0]
                                y2=last[2]

                                try:
                                    if (second_last==0):
                                        y11=0
                                        y21=0
                                    else:
                                        y11=second_last[0]
                                        y21=second_last[2]
                                except:
                                    y11=second_last[0]
                                    y21=second_last[2]

                                #修正追蹤不見的問題
                                if (len(ccount[1])==1) & ((y1+(y2-y1)/2)>count_line):
                                    count_pig-=1

                                if ((y1+(y2-y1)/2)>count_line) & ((y11+(y21-y11)/2)<count_line):
                                    count_pig+=1
                                elif ((y1+(y2-y1)/2)>count_line) & ((y11+(y21-y11)/2)>count_line):
                                    count_pig+=0
                                elif ((y1+(y2-y1)/2)<count_line) & ((y11+(y21-y11)/2)>count_line):
                                    count_pig-=1
                                elif ((y1+(y2-y1)/2)<count_line) & ((y11+(y21-y11)/2)<count_line):
                                    count_pig+=0
                                
                        else:
                            a=a+1
                            #last value
                            last=ccount[1][-1]
                            try:
                                second_last=ccount[1][-2]
                            except:
                                second_last=0

                            y1=last[0]
                            y2=last[2]

                            try:
                                if (second_last==0):
                                    y11=0
                                    y21=0
                                else:
                                    y11=second_last[0]
                                    y21=second_last[2]
                            except:
                                y11=second_last[0]
                                y21=second_last[2]

                            if ((y1+(y2-y1)/2)>count_line) & ((y11+(y21-y11)/2)<count_line):
                                count_pig+=1
                            elif ((y1+(y2-y1)/2)>count_line) & ((y11+(y21-y11)/2)>count_line):
                                count_pig+=0
                            elif ((y1+(y2-y1)/2)<count_line) & ((y11+(y21-y11)/2)>count_line):
                                count_pig-=1
                            elif ((y1+(y2-y1)/2)<count_line) & ((y11+(y21-y11)/2)<count_line):
                                count_pig+=0
                            
                            
                    #tracking-------------------------------
                    save2.append(save)
                    
            print("豬隻總數量:{}".format(str(count_pig)))
            cv2.line(im0s, (count_line,0), (count_line,1080), (0,0,255), 7)
            cv2.rectangle(im0s, (1520,1000), (1920, 1080), (255, 204, 255), -1)
            cv2.putText(im0s, 'pig counts:{}'.format(count_pig), (1520, 1050), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fps=30
                w=im0s.shape[1]
                h=im0s.shape[0]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
            vid_writer.write(im0s)
        vid_writer.release()

        
count_pig_v2(model=model)
torch.cuda.empty_cache()


# In[13]:


torch.cuda.empty_cache()


# In[ ]:




