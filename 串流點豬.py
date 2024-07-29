# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:28:37 2023

@author: ATRI
"""

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
# from sort import *

# from collections import OrderedDict
# from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
# history = OrderedDict()
# class BYTETrackerArgs:
#     track_thresh: float = 0.6
#     track_buffer: int = 30
#     match_thresh: float = 0.8
#     aspect_ratio_thresh: float = 3.0
#     min_box_area: float = 1.0
#     mot20: bool = False
# byte_tracker = BYTETracker(BYTETrackerArgs())



weights = r'C:\Users\ATRI\Desktop\auto_pig\best_multi.pt'
# device = select_device('0')
device = select_device('0')
model = attempt_load(weights, map_location=device)

def count_pig_v2(model=model):
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    half = device.type != 'cpu'
    if half:
        model.half()

    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    names = model.module.names if hasattr(model, 'module') else model.names

    # source='rtsp://admin:Foxconn_88@59.125.76.241:5541/Streaming/Channels/101?transportmode=unicast&profile=Profile_101'
    # source='rtsp://root:Admin1234@59.125.76.241:5540/live1s1.sdp'
    source = r"C:\Users\ATRI\Desktop\auto_pig\0526_4k.mp4"
    # source = '/content/drive/MyDrive/car_detect/count_pig_0510/0526_4k.mp4'
    # dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    start=datetime.datetime.now()
    svefile = datetime.datetime.strftime(start,'%Y-%m-%d %H:%M:%S').replace('-','_')
    svefile = svefile.replace(':','_') 
    svefile = svefile.replace(' ','_')+'.mp4'
    save_path = r'C:\Users\ATRI\Desktop\auto_pig\save\{}'.format(svefile)
    
    vid_path, vid_writer = None, None
    p_count = 0
    start=datetime.datetime.now()

    
    with torch.inference_mode():
        check_list=[]
        check_time=[]
        count_pig=0
        start_count=0
        count_line=750
        save2=[]
        a=0
        for path, img, im0s, vid_cap in tqdm(dataset):

            # use_time=(datetime.datetime.now()-start)
            # print("目前花費時間為 : ",use_time)
            # if use_time >= datetime.timedelta(minutes=1):
            #   break

            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.60, 0.40, agnostic=False)
            # print(check_list)
            # if (len(check_list)>90) & (sum(check_list[-90:]))==0:
            #     print(check_list)
            #             # check_list=[]
            #     break

            for j, det in enumerate(pred):
                # im0s=im0s[j].copy()
                # im0s=im0s[j][200:,:].copy()
                # im0s=im0s[200:,:]
                if len(det):
                    # print(det)
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    check_list.append(int((det[:,-1]==0).sum()))

                    for x1,y1,x2,y2,conf,detclass in det[det[:,-1]==2].cpu().detach().numpy():
                        cv2.rectangle(im0s, (int(x1),int(y1)),(int(x2),int(y2)), (0, 0, 255), 5)
                        
                    for x1,y1,x2,y2,conf,detclass in det[det[:,-1]==0].cpu().detach().numpy():
                        cv2.rectangle(im0s, (int(x1),int(y1)),(int(x2),int(y2)), (255, 0, 0), 5)
            
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                # fps=30
                # w=im0s.shape[1]
                # h=im0s.shape[0]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
            vid_writer.write(im0s)
        vid_writer.release()
        # print("豬豬總數量:{}".format(str(count_pig)))
        
start=datetime.datetime.now()
count_pig_v2(model=model)
print("目前花費時間為 : ",(datetime.datetime.now()-start))

