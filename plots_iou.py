import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})

df=pd.read_csv('model1008_test_iou.csv')

df['iou']=round(df['iou'],1)
df['P']=round(df['P'],3)
df['R']=round(df['R'],3)
df['F1-score']=round((2*df['R']*df['P'])/(df['P']+df['R']),3)
df['mAP@.5']=round(df['mAP@.5'],3)
df['mAP@.5:.95']=round(df['mAP@.5:.95'],3)



# def plot_iou(column,interval):
#     plt.figure(figsize=(8, 6))


#     # 獲取唯一的 'Class' 值，用來分配不同的 marker
#     unique_classes = df['Class'].unique()
#     markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']  # 可以用的 marker 樣式

#     # 使用不同的 marker 根據 'Class' 畫圖
#     sns.lineplot(data=df, 
#                  x="iou", 
#                  y=column,
#                  hue='Class',
#                  style='Class',  # 根據 'Class' 改變 marker
#                  markers=markers[:len(unique_classes)],  # 確保 marker 數量與 'Class' 數量匹配
#                  markersize=10
#                  )  # 去掉陰影

#     # 設定 y 軸範圍和間隔
# #     interval=0.02
#     plt.ylim(min(df[column])-interval, max(df[column])+interval)
#     plt.yticks(np.arange(min(df[column])-interval, max(df[column])+interval, interval))
#     plt.title('{} on different iou'.format(column))
#     # 移動圖例到圖表外側，避免遮擋圖表內容
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#     plt.tight_layout()

#     # 顯示圖表
#     plt.show()

# plot_iou('F1-score',0.02)
# plot_iou('mAP@.5:.95',0.02)
# plot_iou('mAP@.5',0.002)


def plot_iou(column, interval, ax):
    # 獲取唯一的 'Class' 值，用來分配不同的 marker
    unique_classes = df['Class'].unique()
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']  # 可以用的 marker 樣式

    # 使用不同的 marker 根據 'Class' 畫圖
    sns.lineplot(data=df, 
                 x="iou", 
                 y=column,
                 hue='Class',
                 style='Class',  # 根據 'Class' 改變 marker
                 markers=markers[:len(unique_classes)],  # 確保 marker 數量與 'Class' 數量匹配
                 markersize=10,
                 ax=ax  # 設定子圖
                 )  
    # 設定 y 軸範圍和間隔
    ax.set_ylim(min(df[column])-interval/2, max(df[column])+interval/2)
    ax.set_yticks(np.arange(min(df[column])-interval/2, max(df[column])+interval/2, interval))
    ax.set_title('{} on different iou'.format(column))
    # 移動圖例到圖表外側，避免遮擋圖表內容
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)



# 建立一個包含三個子圖的 figure，設定橫向排列
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# 繪製三張圖
plot_iou('F1-score', 0.02, axs[0])
plot_iou('mAP@.5', 0.002, axs[1])
plot_iou('mAP@.5:.95', 0.02, axs[2])
plt.savefig('iou_plots.png', dpi=600)
plt.tight_layout()
plt.show()
