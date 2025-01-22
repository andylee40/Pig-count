---
title: 影像辨識 - 牧場豬隻數量清點
tags: 工作整理
description: 影像辨識
---

# 利用深度學習於牧場豬隻數量清點

## 簡介：
訓練YOLOv7模型偵測豬隻，並結合ByteTrack演算法追蹤偵測目標移動軌跡。最後將此系統部署至現場，藉由現場燈板即時顯示模型計算豬隻的總數量。


## 工具使用介紹：

:point_right:深度學習 : YOLOv7、ByteTrack



## 影片展示：

<video src="https://github.com/user-attachments/assets/800a5ef7-28e6-43c0-ae08-a489cb59b9cd" controls muted=true autoplay=true width=100%></video>

## 檔案存放
* YOLOv7與Bytetrack存放在同層目錄下
* 與原程式碼相異處：
    1. 該專案更改YOLOv7的utils中的 metrics.py，使混淆矩陣呈現為數字，非比例<br>
    2. 該專案更改Bytetrack中些許套件引入路徑<br>
* 新增程式碼介紹：
    1. reshuffle_data.py：若需重新切分資料，使用此檔案<br>
    2. gridsearch_iou.py：計算yolov7偵測時的最佳iou<br>
    3. plots_iou.py：針對不同iou下的模型效能畫圖<br>
    4. autolabel.py：利用訓練後模型自行標註新影片或影像<br>
    5. pigcount_20241126_v4.py：點豬計數主程式碼
