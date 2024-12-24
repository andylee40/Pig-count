import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np 
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel




weights='/home/u7412932/pigcount_0819/yolov7/runs/train/pigcount_guodong_1008/weights/best.pt'
video_path = '/home/u7412932/pigcount_0819/yolov7/20240930_095318_output_back1m.mp4'
absolute_path=os.getcwd()

#--------------------------Load model and setting--------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1
#--------------------------Load model and setting--------------------------------------------


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


def inference(input_image,frame_id):
    global pred_folder

    img0=input_image.copy()
    img = letterbox(img0, 640, stride=32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.565, 0.60, agnostic=False)

    pred_folder=os.path.join(absolute_path,'pred_result_1018')
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    image_folder=os.path.join(pred_folder,'img')
    txt_folder=os.path.join(pred_folder,'label')

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0=img0.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # frame_id='0'
        img_path=os.path.join(image_folder,str(frame_id))
        txt_path=os.path.join(txt_folder,str(frame_id))

        ##代表有偵測到物件
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #cls car:0,people:1,pig:2
                line = (cls, *xywh)  
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
        cv2.imwrite(img_path+ '.jpg', im0)
#--------------------------Load video and inference--------------------------------------------
# # 讀取影片
# cap = cv2.VideoCapture(video_path)

# frame_count = 0
# save_count = 0

# # 檢查影片是否成功打開
# if not cap.isOpened():
#     print("無法打開影片")
#     exit()

# # 取得影片的幀率 (FPS) 和總幀數
# fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # 計算每秒保存的幀數
# desired_images = 2000
# interval = total_frames // desired_images

# while cap.isOpened():
#     # 讀取影片的每一幀
#     ret, frame = cap.read()
    
#     # 如果影片讀完或出錯，退出循環
#     if not ret:
#         break
    
#     # 每interval幀保存一次圖片
#     if frame_count % interval == 0:

#         frame_id=save_count
#         try:
#             inference(frame,frame_id)
#             print('{} ok'.format(frame_id))
#         except:
#             print('{} error'.format(frame_id))

#         save_count += 1
    
#     # 更新幀計數器
#     frame_count += 1

#     # 如果已經存夠 300 張圖片，則停止
#     if save_count >= desired_images:
#         break

# # 釋放影片對象
# cap.release()
# cv2.destroyAllWindows()
# print('處理完成')

# 讀取影片
cap = cv2.VideoCapture(video_path)

frame_count = 0
save_count = 0

# 檢查影片是否成功打開
if not cap.isOpened():
    print("無法打開影片")
    exit()

# 取得影片的幀率 (FPS) 和總幀數
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 設定要保存的最大圖片數量
desired_images = 2000
interval = 5  # 每5幀保存一次圖片

while cap.isOpened():
    # 讀取影片的每一幀
    ret, frame = cap.read()

    # 如果影片讀完或出錯，退出循環
    if not ret:
        break

    # 每5幀進行一次判別和保存
    if frame_count % interval == 0:
        frame_id = save_count
        try:
            inference(frame, frame_id)  # 假設 inference 是你的判別函數
            print('{} ok'.format(frame_id))
        except:
            print('{} error'.format(frame_id))

        save_count += 1

    # 更新幀計數器
    frame_count += 1

    # 如果已經存夠指定數量的圖片，則停止
    if save_count >= desired_images:
        break

# 釋放影片對象
cap.release()
cv2.destroyAllWindows()
print('處理完成')

# 寫入label文件
# map_dict={0:'car',1:'people',2:'pig'}
lines = ["car", "people", "pig"]
label_txt=os.path.join(pred_folder,'label.txt')
# 打開一個文件，模式 'w' 表示寫入，如果文件不存在會自動創建
with open(label_txt, 'w') as f:
    # 將每一行寫入文件，並且加上換行符號 '\n'
    for line in lines:
        f.write(line + '\n')
#--------------------------Load video and inference--------------------------------------------