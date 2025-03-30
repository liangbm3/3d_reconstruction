# ---------------------辅助函数------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 打印鼠标点击位置的HSV值
def print_hsv(image):
    HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    def getpos(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
            print(HSV[y,x])

    cv2.namedWindow("imageHSV", cv2.WINDOW_NORMAL)  # 允许调整窗口大小
    cv2.resizeWindow("imageHSV", 800, 600)  # 设置窗口大小为 800x600
    cv2.imshow("imageHSV", HSV)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # 允许调整窗口大小
    cv2.resizeWindow("image", 800, 600)  # 设置窗口大小为 800x600
    cv2.imshow('image', image)

    cv2.setMouseCallback("imageHSV",getpos)
    cv2.waitKey(0)

# 翻转图像
def flip_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 水平翻转（左右镜像）
    img_horiz = cv2.flip(img, 1)

    # 垂直翻转（上下镜像）
    img_vert = cv2.flip(img, 0)

    # 水平+垂直翻转（180° 旋转）
    img_both = cv2.flip(img, -1)

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Horizontal Flip", img_horiz)
    cv2.imshow("Vertical Flip", img_vert)
    cv2.imshow("Both Flip", img_both)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
# z轴方向的线性插值
def linear_interpolation(images,factor,angle):

    num_original = len(images)  # 原始图片数量
    num_target = num_original*factor # 目标图片数量

    # 计算插值目标索引
    original_indices = np.linspace(0, num_original - 1, num_original)  # 原始索引
    target_indices = np.linspace(0, num_original - 1, num_target)  # 目标索引

    expanded_images = []
    for idx in target_indices:
        print(idx)
        lower_idx = int(np.floor(idx))
        upper_idx = int(np.ceil(idx))

        if lower_idx == upper_idx:  # 如果目标索引正好是整数
            interpolated_image = images[lower_idx]
        else:
            I1 = images[lower_idx].astype(np.float32)
            I2 = images[upper_idx].astype(np.float32)
            interpolated_image = I1 + (idx - lower_idx) / (upper_idx - lower_idx) * (I2 - I1)
            interpolated_image = np.clip(interpolated_image, 0, 255).astype(np.uint8)  # 归一化处理
        expanded_images.append(interpolated_image)

    # 保存插值后的图片
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    for i, img in enumerate(expanded_images):
        save_path = os.path.join(output_folder, f"interpolated_{angle}_{i:04d}.png")
        cv2.imwrite(save_path, img)

    print(f"插值完成，已生成 {len(expanded_images)} 张图片，存储于 {output_folder}")
    