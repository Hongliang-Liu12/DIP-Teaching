import numpy as np
import cv2

np.set_printoptions(threshold=np.inf)


def create_mask_from_points(points, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)  # 初始化黑色背景的掩码
    points = points.reshape((-1, 1, 2)).astype(np.int32)  # 转换点的格式
    
    # 使用 cv2.fillPoly 填充多边形
    cv2.fillPoly(mask, [points], 255)
    
    return mask

# 示例：创建一个简单的多边形掩码
points = np.array([[2, 2], [4, 2], [4, 4], [2, 4]])  # 正方形
img_h, img_w = 10, 10
# 创建掩码
mask = create_mask_from_points(points, img_h, img_w)

print(mask)
# 显示掩码
