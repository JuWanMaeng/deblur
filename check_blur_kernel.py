import numpy as np
import matplotlib.pyplot as plt
import random,cv2,glob
from tqdm import tqdm
import os
from os import path as osp

numpys=os.listdir('/raid/joowan/UFP_numpy')
numpys.sort()


print(len(numpys))

odd_indexed_values = numpys[1::2]
print(len(odd_indexed_values))


for kernel in tqdm(odd_indexed_values):

    path='/raid/joowan/UFP_numpy/' + kernel
    blur_kernels=np.load(path)
    img_num=kernel.split('_')[0]
    # print(blur_kernels.shape)
    blur_kernel=blur_kernels[0]

    dir_name='/raid/joowan/blur_kernel_results'
    

    save_folder=osp.join(dir_name,img_num)
    if not os.path.exists(save_folder):
        # Create the directory
        os.mkdir(save_folder)


    
    for i in range(0,720):
        for j in range(0,1280):
            a_blur_kernel=blur_kernel[:,i,j]
            a_blur_kernel= a_blur_kernel*255.0
            a_blur_kernel=a_blur_kernel.astype(np.uint8)
            a_blur_kernel=a_blur_kernel.reshape(19,19)
    


            cv2.imwrite(f'{save_folder}/{i}_{j}.png',a_blur_kernel)
        
            
 

        
            
 
