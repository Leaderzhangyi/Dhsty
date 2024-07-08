import cv2
import numpy as np

# 读取图像
image = cv2.imread('datasets/dh/testB/0.jpg')

# 将图像从BGR转换为HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色和绿色的HSV范围
# 红色可能跨越两个区间
"""
0 是色相（Hue）的下限值。色相的范围是 0 到 180，表示颜色的种类。0 通常表示红色。
30 是饱和度（Saturation）的下限值。饱和度的范围是 0 到 255，表示颜色的纯度或强度。较低的饱和度值表示颜色较淡或灰度。
60 是亮度（Value）的下限值。亮度的范围是 0 到 255，表示颜色的明亮程度。较低的亮度值表示颜色较暗。
"""
# 定义HSV范围
# lower_hsv = np.array([13, 48, 173])  # 色相减5，饱和度减10，亮度减10
# upper_hsv = np.array([23, 68, 193])  # 色相加5，饱和度加10，亮度加10
lower_red1 = np.array([0, 30, 60])
upper_red1 = np.array([20, 255, 255])
lower_red2 = np.array([156, 30, 60])
upper_red2 = np.array([180, 255, 255])

# 绿色的HSV范围
lower_green = np.array([35, 30, 60])
upper_green = np.array([85, 255, 255])

# 创建掩模
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # 合并两个红色掩模

mask_green = cv2.inRange(hsv, lower_green, upper_green)

# 提取红色和绿色部分
red_result = cv2.bitwise_and(image, image, mask=mask_red)
green_result = cv2.bitwise_and(image, image, mask=mask_green)


# 保存图片
cv2.imwrite('Original Image.jpg', image)
cv2.imwrite('red_result.jpg', red_result)
cv2.imwrite('green_result.jpg', green_result)

# # 显示结果
# cv2.imshow('Red Parts', red_result)
# cv2.imshow('Green Parts', green_result)
