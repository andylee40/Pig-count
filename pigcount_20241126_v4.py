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


from argument import config
#引數設置
track_thresho=config.track_thresho
confi_thresh=config.confi_thresh
iou_thresh=config.iou_thresh
weights=config.weights
source=config.source

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
    track_thresh: float = track_thresho
    #軌跡保留幀數
    track_buffer: int = 30
    #iou match 越低越嚴格
    match_thresh: float = 0.8
    #目標長寬比的閾值
    # aspect_ratio_thresh: float = 3.0
    #目標面積的閾值
    min_box_area: float = 1.0
    #不使用mot20資料進行測試
    mot20: bool = False
    
byte_tracker = BYTETracker(BYTETrackerArgs())


def process_counts(last, second_last, count_line):

    ##回傳資料格式為(xyxy)
    y1, y2 = last[1], last[3]
    
    # 檢查 second_last 是否是一個有效的數組，且數組長度不為 0
    if isinstance(second_last, np.ndarray) and len(second_last) > 0:
        y11, y21 = second_last[1], second_last[3]
    else:
        y11, y21 = 0, 0

    if (y1 + (y2 - y1) / 2) > count_line and 0<(y11 + (y21 - y11) / 2) < count_line:
        return 1
    elif 0<(y1 + (y2 - y1) / 2) < count_line-50 and (y11 + (y21 - y11) / 2) > count_line-50:
        return -1
    return 0


# 主函數
def count_pig_v2(model, source):
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    half = device.type != 'cpu'
    if half:
        model.half()

    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    start = datetime.datetime.now()
    svefile = datetime.datetime.strftime(start, '%Y-%m-%d %H:%M:%S').replace('-', '_').replace(':', '_').replace(' ', '_') + 'predict.mp4'
    save_path = os.path.join(os.getcwd(), svefile)
    vid_path, vid_writer = None, None
    count_pig = 0
    a = 0

    with torch.inference_mode():
        for path, img, im0s, vid_cap in tqdm(dataset):

            y_height=im0s.shape[0]
            x_width=im0s.shape[1]
            count_line=int(y_height/2)+50

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, confi_thresh, iou_thresh, agnostic=False)

            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    pig_id = 1
                    for x1, y1, x2, y2, conf, detclass in det[det[:, -1] == pig_id].cpu().detach().numpy():
                        cv2.rectangle(im0s, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

                    tracks = byte_tracker.update(
                        output_results=det[det[:, -1] == pig_id][:, :5].cpu(),
                        img_info=im0s.shape,
                        img_size=im0s.shape
                    )

                    updated_ids = set()
                    for track in tracks:
                        xyxy = track.tlbr
                        class_id = track.track_id
                        frame_id = track.frame_id
                        his_info = np.hstack([xyxy, np.array([frame_id])])

                        if class_id in history:
                            history[class_id].append(his_info)
                            # 限制歷史記錄長度
                            if len(history[class_id]) > 100:  
                                history[class_id] = history[class_id][-100:]
                        else:
                            history[class_id] = [his_info]
                        updated_ids.add(class_id)

                    for class_id, track_info in history.items():

                        # 計算豬隻數量
                        last = track_info[-1]
                        second_last = track_info[-2] if len(track_info) > 1 else 0
                        count_pig += process_counts(last, second_last, count_line)

                        # 未更新的 class_id不畫軌跡與ID
                        if class_id not in updated_ids:
                            continue
                        
                        # 畫物件ID
                        x1, y1, x2, y2 = map(int, last[:4]) 
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # 計算中心點
                        cv2.putText(im0s, f'ID: {class_id}', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

                        # 畫物件軌跡
                        if len(track_info) > 1:
                            for i in range(1, len(track_info)):
                                x1, y1, x2, y2 = track_info[i - 1][:4]
                                x1_next, y1_next, x2_next, y2_next = track_info[i][:4]
                                center_prev = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                center_next = (int((x1_next + x2_next) / 2), int((y1_next + y2_next) / 2))
                                cv2.line(im0s, center_prev, center_next, (255, 255, 0), thickness=2)
                                cv2.circle(im0s, center_next, radius=3, color=(0, 255, 0), thickness=-1)

            print("豬隻總數量:{}".format(str(count_pig)))
            # cv2.line(im0s, (0,count_line-50), (x_width,count_line-50), (255,0,0), 6) ##橫線
            cv2.line(im0s, (0,count_line), (x_width,count_line), (0,0,255), 6) ##橫線
            cv2.rectangle(im0s, (x_width-400,y_height-80), (x_width, y_height), (255, 204, 255), -1)
            cv2.putText(im0s, 'pig counts:{}'.format(count_pig), (x_width-400,y_height-30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)

            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fps = 30
                w, h = im0s.shape[1], im0s.shape[0]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
            vid_writer.write(im0s)
        vid_writer.release()

count_pig_v2(model=model, source=source)
torch.cuda.empty_cache()

