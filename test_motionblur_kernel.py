from motionblur.motionblur import Kernel
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from numpy.random import uniform, triangular, beta
from math import pi
from pathlib import Path
from scipy.signal import convolve



scale=0.5
s=19

# 블러 커널 생성 코드
for i in range(100):
    kernel=Kernel(size=(s,s),intensity=scale)
    kernel.displayKernel(save_to=f'hdd/motionblur_kernels/19x19/motion_blur_kernel_{i}.png',show=True)


# 블러 커널 생성후 저장한다음 이미지로 불러와서  블러를 적용하는 코드

# kernel = Kernel(size=(100,100), intensity=1)
# kernel.displayKernel(save_to="hdd/motionblur_results/kernel.png", show=False)
# image= Image.open("UFPDeblur/datasets/GoPro/test/sharp/000001.png")


# blurred1 = kernel.applyTo(image)
# blurred1.save('hdd/motionblur_results/blurred.png')

# blurred_same = kernel.applyTo(image, keep_image_dim=True)
# blurred_same.save('hdd/motionblur_results/blurred_same.png')




# 직접 블러 적용하는 코드

# # sharp image
# img=Image.open('UFPDeblur/datasets/GoPro/test/sharp/000001.png')

# # blur_kernel
# blur_kernel=Image.open('hdd/motionblur_kernels/19x19_1_3.png')
# kernel = np.asarray(blur_kernel, dtype=np.float32)
# kernel /= np.sum(kernel)

# print(kernel.size)

# result_bands=[]

# for band in img.split():
#     result_band=convolve(
#         band, kernel, mode='same').astype('uint8')

    
#     result_bands.append(result_band)

# result=np.dstack(result_bands)

# res=Image.fromarray(result)
# print(res.size)
# res.save('hdd/motionblur_results/res.png')

