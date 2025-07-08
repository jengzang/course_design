import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像为灰度图
img = cv2.imread('C:/', 0)

# 傅里叶变换
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(dft)

# 创建掩模（低通滤波器）
rows, cols = img.shape
crow, ccol = rows // 2 , cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# 应用掩模
fshift = fshift * mask

# 逆变换
ishift = np.fft.ifftshift(fshift)
iimg = cv2.idft(ishift)
res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

# 显示原图和滤波后的图像
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(res, cmap='gray'), plt.title('Result Image')
plt.axis('off')
plt.show()
