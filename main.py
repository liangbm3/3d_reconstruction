import cv2
import numpy as np
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import open3d as o3d
import scipy.ndimage as ndimage
import os

#------------------------索引提取------------------------

def compute_sharpness(image):
    """计算图像的清晰度（Laplacian 方差）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    
    # 使用窗口均值计算 Laplacian 变化率（降低噪声）
    window_size = 5  # 窗口大小
    laplacian = cv2.boxFilter(laplacian, ddepth=-1, ksize=(window_size, window_size))
    
    return laplacian


angles = [0, 1, 2, 3]

nums = [73, 78, 98, 94]

nums=[x*6 for x in nums]

for(angle, num) in zip(angles, nums):
    print("提取索引")
    # 获取图片
    images = [cv2.imread(f"./outputs/interpolated_{angle}_{i:04d}.png") for i in range(num)]

    # 获取图像尺寸
    H, W, C = images[0].shape

    # 计算每张图片的清晰度图（使用Laplacian算子）
    sharpness_stack = np.array([compute_sharpness(img) for img in images])  # (N, H, W)

    # 高斯滤波去噪
    # sharpness_stack = np.array([cv2.GaussianBlur(sh, (5, 5), 0) for sh in sharpness_stack])

    # 计算每个像素点最清晰的焦点索引
    best_focus_indices = np.argmax(sharpness_stack, axis=0)  # (H, W)
    # print("清晰度图:",best_focus_indices.shape)
    
    # 超像素平滑
    # best_focus_indices = ndimage.median_filter(best_focus_indices, size=5)  # 中值滤波去除噪点
    

    # 堆叠图像，生成全焦点图像
    focus_images_stack = np.stack(images, axis=0)  # (N, H, W, C)
    all_in_focus = np.take_along_axis(focus_images_stack, best_focus_indices[None, ..., None], axis=0).squeeze(0)

    # ------------------ 物体分割 ------------------
    print("物体分割")
    # 转换到HSV空间，并进行颜色范围选择（例如：检测棕色物体）
    hsv_focus = cv2.cvtColor(all_in_focus, cv2.COLOR_BGR2HSV)  # 转换为HSV颜色空间
    lower_brown, upper_brown = np.array([0, 0, 50]), np.array([40, 150, 255])  # 棕色的HSV范围
    mask = cv2.inRange(hsv_focus, lower_brown, upper_brown)  # 创建颜色掩码

    # 形态学操作去噪
    kernel = np.ones((5, 5), np.uint8)  # 定义一个5x5的卷积核
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算（去除小噪声）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算（填补小孔）

    # 边缘检测并增强掩码
    edges = cv2.Canny(mask, 100, 200)  # 使用Canny边缘检测
    mask = cv2.bitwise_or(mask, edges)  # 合并边缘和掩码

    # 查找轮廓并生成最终的物体掩码
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找外部轮廓
    final_mask = np.zeros_like(mask)  # 创建一个与mask相同大小的空白图像


    cv2.drawContours(final_mask, contours, -1, (255), thickness=cv2.FILLED)  # 填充轮廓区域

    #-------------------- 图像显示 ------------------
    
    print("显示图像")
    
    # 叠加物体掩码
    all_in_focus = cv2.bitwise_and(all_in_focus, all_in_focus, mask=final_mask)  # 仅保留物体区域

    # 保存全焦点图像
    cv2.imwrite(f"focus_{angle}.jpg", all_in_focus)
    # cv2.imshow("All in Focus", all_in_focus)

    # 像素伪彩色深度图
    min_depth = np.min(best_focus_indices)
    max_depth = np.max(best_focus_indices)
    depth_map = (best_focus_indices - min_depth) / (max_depth - min_depth)  # 归一化到[0, 1]
    depth_map=1-depth_map
    depth_map = (depth_map * 255).astype(np.uint8)  # 转换为8位图像
    colored_depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)  # 伪彩色化

    #叠加物体掩码
    colored_depth_map = cv2.bitwise_and(colored_depth_map, colored_depth_map, mask=final_mask)  # 仅保留物体区域

    cv2.imwrite(f"depth_map_{angle}.jpg", colored_depth_map)
    # cv2.imshow("Depth Map", colored_depth_map)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ------------------ 深度图转换为点云 -----------------
    
    best_focus_indices = best_focus_indices.astype(np.float32)
    # print("深度图:",best_focus_indices)
    # best_focus_indices = cv2.bilateralFilter(best_focus_indices, 9, 75, 75)  # 使用双边滤波平滑深度图
    # print("双边滤波:",best_focus_indices.shape)
    # print(best_focus_indices)
    
    # best_focus_indices = best_focus_indices[::2, ::2]  # 降采样 
    
    # print("大小：",best_focus_indices.shape)
    best_focus_indices = ndimage.median_filter(best_focus_indices, size=11)  # 中值滤波去除噪点
    best_focus_indices = cv2.GaussianBlur(best_focus_indices, (31,31), 0)
    # print("高斯滤波：",best_focus_indices.shape)
    # print(best_focus_indices)
    print("深度图转换为点云")

    # 定义相机内参（假设焦距 fx, fy 和主点位置 cx, cy）
    fx, fy, cx, cy = 400, 400, W // 2, H // 2

  # 计算像素的归一化坐标（确保索引方式正确）
    k_indices, j_indices = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    # 计算深度值
    z_values = best_focus_indices * 0.5 + 80  # 深度值计算

    # 计算三维坐标
    x_values = (j_indices - cx) * 80 / fx
    y_values = (k_indices - cy) * 80 / fy

    # 生成有效点掩码，确保 final_mask 形状与 best_focus_indices 一致
    valid_mask = final_mask > 0

    # 筛选有效点
    x_selected = x_values[valid_mask]
    y_selected = y_values[valid_mask]
    z_selected = z_values[valid_mask]
    points = np.stack((x_selected, y_selected, z_selected), axis=-1)

    # 归一化颜色信息（转换为 RGB 格式并归一化）
    colors = colored_depth_map[valid_mask] / 255.0  # BGR 转 RGB
    colors = colors[:, ::-1]  # 交换 B 和 R 以匹配 RGB

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 赋值点
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 赋值颜色

    # 显示点云
    o3d.visualization.draw_geometries([pcd], window_name="Colored Point Cloud")
    os.makedirs("./point_cloud", exist_ok=True)
    # 保存点云
    o3d.io.write_point_cloud(f"./point_cloud/point_cloud_{angle}.ply", pcd)  # 保存为PLY格式
