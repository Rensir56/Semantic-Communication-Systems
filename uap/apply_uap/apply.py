import cv2
import numpy as np

# 加载彩色 UAP
uap = cv2.imread('uap_single.png')  # 彩色 UAP
uap = uap.astype(np.float32)  # 转换为浮点数

# 加载目标图片
target_image = cv2.imread('bengin.jpg')  # 目标图片
target_image = target_image.astype(np.float32)  # 转换为浮点数

# 检查 UAP 和目标图片的尺寸是否匹配
if uap.shape != target_image.shape:
    print("UAP 和目标图片的尺寸不匹配，需要调整 UAP 的尺寸。")
    uap = cv2.resize(uap, (target_image.shape[1], target_image.shape[0]))  # 调整 UAP 尺寸

# 归一化 UAP
uap = uap / 255.0  # 将 UAP 的像素值从 [0, 255] 缩放到 [0, 1]

# 控制扰动的强度
epsilon = 0.05 # 可调整该参数以控制扰动的强度
uap = uap * epsilon

# 将 UAP 添加到目标图片上（分别对每个通道处理）
adversarial_image = target_image + uap

# 确保像素值在 [0, 255] 范围内
adversarial_image = np.clip(adversarial_image, 0, 255)
adversarial_image = adversarial_image.astype(np.uint8)  # 转换回整数类型

# 保存生成的对抗样本
cv2.imwrite('对抗样本.jpg', adversarial_image)