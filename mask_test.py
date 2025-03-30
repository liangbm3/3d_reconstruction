#---------------------掩码测试---------------------#

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_shell_mask(image_path):
    # 读取图像并预处理
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以便 matplotlib 显示
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.figure()
    plt.imshow(hsv)
    # 定义贝壳颜色范围（根据实物调整）
    lower_brown = np.array([0, 0, 50])
    upper_brown = np.array([40, 150, 255])
    
    # 创建颜色掩码
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # 形态学去噪（开运算消除噪点，闭运算填充孔洞）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 边缘检测增强
    edges = cv2.Canny(mask, 100, 200)
    mask = cv2.bitwise_or(mask, edges)
    
    # 查找轮廓并生成最终掩码
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # 提取前景
    foreground = cv2.bitwise_and(img, img, mask=final_mask)
    foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以便 matplotlib 显示
    
    return img_rgb, mask, final_mask, foreground_rgb

# 使用示例
image_path = 'all_in_focus_angle_3.jpg'
original, mask, final_mask, foreground = extract_shell_mask(image_path)

# 使用 matplotlib 可视化结果
plt.figure(figsize=(12, 8))

# 原始图像
plt.subplot(2, 2, 1)
plt.imshow(original)
plt.title('Original Image')
plt.axis('off')

# 颜色掩码
plt.subplot(2, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Color Mask')
plt.axis('off')

# 最终掩码
plt.subplot(2, 2, 3)
plt.imshow(final_mask, cmap='gray')
plt.title('Final Mask')
plt.axis('off')

# 提取的前景
plt.subplot(2, 2, 4)
plt.imshow(foreground)
plt.title('Foreground Extracted')
plt.axis('off')

plt.tight_layout()
plt.show()