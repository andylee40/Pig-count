## :fire: 簡介 
利用yolov7先框出目標，再利用Bytetrack追蹤目標軌跡，達到目標計數

<br>

## 檔案存放
yolov7與Bytetrack存放在同層目錄下


<br>

## 與原程式碼相異處
1. 該專案更改yolov7的utils中的metrics.py，使混淆矩陣呈現為數字，非比例<br>
2. 該專案更改Bytetrack中些許套件引入路徑<br>

<br>

## 新增程式碼介紹
* reshuffle_data.py：若需重新切分資料，使用此檔案<br>
* gridsearch_iou.py：計算yolov7偵測時的最佳iou<br>
* plots_iou.py：針對不同iou下的模型效能畫圖<br>
* autolabel.py：利用訓練後模型自行標註新影片或影像<br>
* pigcount_20241126_v4.py：點豬計數主程式碼
