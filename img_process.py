# -------------------图片预处理-------------------

from functions import *
import cv2
import os

angles = [0, 1, 2, 3]
nums = [73, 78, 98, 94]
factor = 0.6  # 缩放因子


os.makedirs("./img2", exist_ok=True)

# 读取图片并进行缩放
for angle, num in zip(angles, nums):
    images = [cv2.imread(f"./img/angle_{angle}_{i:06d}.jpg") for i in range(num)]
    for i, img in enumerate(images):
        if img is None:  # 检查图片是否成功读取
            print(f"Warning: Failed to read ./img/angle_{angle}_{i:06d}.jpg")
            continue
        resized_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
        output_path = f"./img2/resize_angle_{angle}_{i:06d}.jpg"
        cv2.imwrite(output_path, resized_img)
        print(output_path)

# 读取缩放后的图片并进行线性插值
for angle,num in zip(angles,nums):
    print(angle,num)
    images = [cv2.imread(f"./img2/resize_angle_{angle}_{i:06d}.jpg") for i in range(num)]
    linear_interpolation(images,6,angle)