# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (three_paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from torch.utils.data import IterableDataset
from PIL import Image,ImageOps
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import gc
import torch.multiprocessing as mp
import torchvision

def load_three_pair_data_dist(opt):  # test

    
    batch_size = opt['batch_size_per_gpu']
    num_workers = opt['num_worker_per_gpu']
    dataset = ThreePairImageDataset(opt)

    
    
    return dataset

def load_pair_data_NAFNet_dist(opt):

    
    random_crop = opt['random_crop']
    random_flip = opt['random_flip']
    batch_size = opt['batch_size_per_gpu']
    num_workers = opt['num_worker_per_gpu']
    dataset = PairImageIterableDatasetNAFNet(
        opt,
        random_crop,
        random_flip
 
    )

    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers)
    
    # sharp_loader = DataLoader(
    #         sharp_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True
    #     )
    # blur_loader = DataLoader(
    #         blur_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True
    #     )
    return data_loader


def load_three_pair_data(opt):  # test

    

    # num_workers = opt['num_worker_per_gpu']
    dataset = ThreePairImageDataset(opt)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    return data_loader

def load_pair_data_NAFNet(opt):

    
    random_crop = opt['random_crop']
    random_flip = opt['random_flip']
    batch_size = opt['batch_size_per_gpu']
    num_workers = opt['num_worker_per_gpu']
    dataset = PairImageIterableDatasetNAFNet_legacy(
        opt,
        random_crop,
        random_flip
 
    )

    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers)
    
    # sharp_loader = DataLoader(
    #         sharp_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True
    #     )
    # blur_loader = DataLoader(
    #         blur_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True
    #     )
    return data_loader

class ThreePairImageDataset(Dataset):
    def __init__(self, opt, random_crop = False):
        super().__init__()

        self.random_crop = random_crop
        self.opt = opt
        # d_opt = opt['datasets'].get('test')
        self.gt_folder, self.lq_folder, self.kernel_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_kernel'] 
        # self.gt_folder, self.lq_folder, self.kernel_folder = d_opt.get('dataroot_gt'),d_opt.get('dataroot_lq'),d_opt.get('dataroot_kernel') 
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

   
        self.paths = three_paired_paths_from_folder(
            [self.lq_folder, self.gt_folder,self.kernel_folder], ['lq', 'gt', 'kernel'],
            self.filename_tmpl)

    def process_image(self, sharp_path, blur_path, kernel_path):
        sharp_image = Image.open(sharp_path)
        sharp_image = sharp_image.convert("RGB")
        sharp_tensor = transforms.ToTensor()(sharp_image)
        # sharp_tensor = sharp_tensor.to(device='cuda')
        C,H,W = sharp_tensor.shape
        
        blur_image = Image.open(blur_path)
        blur_image = blur_image.convert("RGB")
        blur_tensor = transforms.ToTensor()(blur_image)
        # blur_tensor = blur_tensor.to(device='cuda')


        kernel_image = Image.open(kernel_path)
        kernel_image = kernel_image.convert('L')
        kernel_tensor = transforms.ToTensor()(kernel_image)
        kernel_tensor = kernel_tensor.to(device='cuda')
        kernel_tensor = replicate_kernel(kernel_tensor,H,W)

        if self.random_crop:
            sharp_tensor, blur_tensor, kernel_tensor = random_crop_pair_tensor(sharp_tensor, blur_tensor, kernel_tensor, size=128)
        

        return sharp_tensor, blur_tensor, kernel_tensor

    def __getitem__(self, index):
        path = self.paths[index]
        gt_path = path['gt_path']
        lq_path = path['lq_path']
        kernel_path = path['kernel_path']
        
        sharp_tensor, blur_tensor, kernel_tensor = self.process_image(gt_path, lq_path , kernel_path)
        

        return {'gt': sharp_tensor, 'lq': blur_tensor, 'kernel': kernel_tensor, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)


class PairedImageKernelDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageKernelDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.kernel_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_kernel']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        ############ 여기부터 시작 ##########
        self.paths = three_paired_paths_from_folder(
            [self.lq_folder, self.gt_folder,self.kernel_folder], ['lq', 'gt', 'kernel'],
            self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class PairImageIterableDatasetNAFNet(IterableDataset):
    def __init__(
        self,
        opt,
        random_crop,
        random_flip
    ):
        super().__init__()

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.kernel_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_kernel']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = three_paired_paths_from_folder(
            [self.lq_folder, self.gt_folder,self.kernel_folder], ['lq', 'gt', 'kernel'],
            self.filename_tmpl)

    def process_image(self, sharp_path, blur_path, kernel_path):
        H, W = 720, 1280
        
        sharp_tensor = torch.zeros((3,768,1280))
        sharp_image = Image.open(sharp_path)
        sharp_image = sharp_image.convert("RGB")
        sharp = transforms.ToTensor()(sharp_image)
        sharp_tensor[:,:H,:W] = sharp
        
        
        blur_tensor = torch.zeros((3,768,1280))
        blur_image = Image.open(blur_path)
        blur_image = blur_image.convert("RGB")
        blur = transforms.ToTensor()(blur_image)
        blur_tensor[:,:H,:W] = blur
        
        if self.random_crop:
            sharp_tensor, blur_tensor, h, w = random_crop_pair_tensor(sharp_tensor, blur_tensor, size=256)

        kernel_image = Image.open(kernel_path)
        kernel_image = kernel_image.convert('L')
        kernel_tensor = transforms.ToTensor()(kernel_image)
        kernel_tensor = preprocess_kernels(kernel_tensor, h, w)
        
       
        sharp_tensor = sharp_tensor.to(dtype=torch.float32)
        blur_tensor = blur_tensor.to(dtype=torch.float32)
        kernel_tensor = kernel_tensor.to(dtype=torch.float32)
         
        return sharp_tensor, blur_tensor, kernel_tensor

    def __iter__(self):
        while True:
            for path in self.paths:
                lq_path = path['lq_path']
                gt_path = path['gt_path']
                kernel_path = path['kernel_path']
            
                sharp_tensor, blur_tensor, kernel_tensor = self.process_image(gt_path,lq_path , kernel_path)
            

                yield {'gt': sharp_tensor, 'lq': blur_tensor, 'kernel': kernel_tensor}
          
                
    def __len__(self):
        return len(self.paths)



def random_crop_pair_tensor(sharp_tensor,blur_tensor,size):

    crop = size
    h = torch.randint(0, 720 - crop , (1,))       # height
    w = torch.randint(0, 1280 - crop , (1,))      # width
    crop_h,crop_w = size, size

    sharp = sharp_tensor[:, h:h+crop_h, w:w+crop_w]
    blur = blur_tensor[:, h:h+crop_h, w:w+crop_w]
    

    return sharp, blur, h, w


def preprocess_kernels(kernel, h, w):
    origin = kernel
    # 영역 크기 (64, 64)
    patch_size = (19,19)

    # 스트라이드 (stride) 크기 (64, 64)
    stride = (19,19)
    kernel_size = 19

    C,H,W = kernel.shape

    row = H//patch_size[0]
    col = W//patch_size[0]
    num_total_patches = row * col

    start_kernel_index = (int(h) // 128) * 10 + (int(w) // 128)

    kernel_index = [start_kernel_index, start_kernel_index+1, start_kernel_index+2,
                    start_kernel_index+10, start_kernel_index+11, start_kernel_index+12,
                    start_kernel_index+20, start_kernel_index+21, start_kernel_index+22]
    num_total_patches = 9
    rr,rc = 3,3
    ra = rr * 128


    kernel = kernel.unfold(1, patch_size[0], stride[0]).unfold(2,patch_size[1],stride[1]).reshape(1,-1,patch_size[0],patch_size[1]).permute(1,0,2,3)  # 60, 1, 19, 19
    kernel = kernel[kernel_index]  # 9,1,19,19

    kernel = kernel.view(num_total_patches,-1)                    # n, 361
    

    index = 0
    a = torch.zeros((ra,ra,361))
    for i in range(0,ra,128):
        for j in range(0,ra,128):
            a[i:i+128,j:j+128,:] = kernel[index, :]
            index +=1

    kernel = a.permute(2,0,1)

    # kernel cropping
    r_offset, c_offset = (int(h) // 128) * 128, (int(w) // 128 ) * 128 
    rel_r, rel_c = int(h) - r_offset, int(w) - c_offset

    kernel = kernel[:,rel_r:rel_r+256, rel_c:rel_c+256]
    
    return kernel  # 361, 384, 384,




class PairImageIterableDatasetNAFNet_legacy(IterableDataset):
    def __init__(
        self,
        opt,
        random_crop,
        random_flip
    ):
        super().__init__()

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.kernel_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_kernel']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = three_paired_paths_from_folder(
            [self.lq_folder, self.gt_folder,self.kernel_folder], ['lq', 'gt', 'kernel'],
            self.filename_tmpl)

    def process_image(self, sharp_path, blur_path, kernel_path):
        sharp_image = Image.open(sharp_path)
        sharp_image = sharp_image.convert("RGB")
        sharp_tensor = transforms.ToTensor()(sharp_image)
        # sharp_tensor = sharp_tensor.to(device='cuda')
        C,H,W = sharp_tensor.shape
        
        blur_image = Image.open(blur_path)
        blur_image = blur_image.convert("RGB")
        blur_tensor = transforms.ToTensor()(blur_image)
        # blur_tensor = blur_tensor.to(device='cuda')



        kernel_image = Image.open(kernel_path)
        kernel_image = kernel_image.convert('L')
        kernel_tensor = transforms.ToTensor()(kernel_image)
        # kernel_tensor = kernel_tensor.to(device='cuda')
        kernel_tensor = replicate_kernel_legacy(kernel_tensor,H,W)

        #######
        # tensor = tensor[:,:,21:40,21:40]
        #######
        
        

        if self.random_crop:
            sharp_tensor, blur_tensor, kernel_tensor = random_crop_pair_tensor_legacy(sharp_tensor, blur_tensor, kernel_tensor, size=256)
        
        return sharp_tensor, blur_tensor, kernel_tensor

    def __iter__(self):
        while True:
            for path in self.paths:
                lq_path = path['lq_path']
                gt_path = path['gt_path']
                kernel_path = path['kernel_path']
            
                sharp_tensor, blur_tensor, kernel_tensor = self.process_image(gt_path,lq_path , kernel_path)
            

                yield {'gt': sharp_tensor, 'lq': blur_tensor, 'kernel': kernel_tensor}
          
                
    def __len__(self):
        return len(self.paths)



def random_crop_pair_tensor_legacy(sharp_tensor,blur_tensor, kernel_tensor,size):

    crop = size
    top = torch.randint(0, 720 - crop + 1, (1,))       # height
    left = torch.randint(0, 1280 - crop + 1, (1,))      # width
    crop_h,crop_w = size, size

    sharp = sharp_tensor[:, top:top+crop_h, left:left+crop_w]
    blur = blur_tensor[:, top:top+crop_h, left:left+crop_w]
    kernel = kernel_tensor[:,top:top+crop_h, left:left+crop_w]
    

    return sharp,blur,kernel


def replicate_kernel_legacy(kernel,orig_H,orig_W):
    # 영역 크기 (64, 64)
    patch_size = (64, 64)

    # 스트라이드 (stride) 크기 (64, 64)
    stride = (64, 64)
    kernel_size = 19

    C,H,W = kernel.shape

    row = H//patch_size[0]
    col = W//patch_size[0]
    num_total_patches = row * col


    kernel = kernel.unfold(1, patch_size[0], stride[0]).unfold(2,patch_size[1],stride[1]).reshape(1,-1,patch_size[0],patch_size[1]).permute(1,0,2,3)
    kernel = F.interpolate(kernel, size=(19, 19), mode='bilinear', align_corners=False)       # 60, 1, 19, 19
    kernel = kernel.view(num_total_patches,-1).unsqueeze(-1).unsqueeze(-1)                    # 60, 361, 1, 1
    kernel = kernel.repeat(1,1,128,128)
    kernel = kernel.reshape(row, col,kernel_size * kernel_size, 128, 128)
    kernel = kernel.permute(0,3,1,4,2)                                    
    kernel = kernel.reshape(768,1280, kernel_size * kernel_size)         # 768, 1280, 361

    # kernels = torch.zeros((768,1280,kernel_size * kernel_size))
    # kernels=kernels.unfold(0,128,128).unfold(1,128,128).reshape(60,kernel_size * kernel_size, 128, 128)

    # kernels[:,:,:,:] = tensor
    # kernels = kernels.reshape(row, col,kernel_size * kernel_size, 128, 128)  # 6, 10, 361, 128, 128


    # kernels = kernels.permute(0, 3, 1, 4, 2)   # 6,128,10,128,361
    # kernels=kernels.reshape(768, 1280, kernel_size * kernel_size)  # 768, 1280, 361
    torch.cuda.empty_cache()

    kernel = kernel[:orig_H,:orig_W,:]
    kernel = kernel.permute(2,0,1)
    
    return kernel

# def replicate_kernel(kernel,orig_H,orig_W):

#     # 영역 크기 (64, 64)
#     patch_size = (64, 64)

#     # 스트라이드 (stride) 크기 (64, 64)
#     stride = (64, 64)
#     kernel_size = 19

#     C,H,W = kernel.shape

#     row = H//patch_size[0]
#     col = W//patch_size[0]
#     num_total_patches = row * col


#     tensor = kernel.unfold(1, patch_size[0], stride[0]).unfold(2,patch_size[1],stride[1]).reshape(1,-1,patch_size[0],patch_size[1]).permute(1,0,2,3)
#     tensor = F.interpolate(tensor, size=(19, 19), mode='bilinear', align_corners=False)       # 60, 1, 19, 19
#     tensor = tensor.view(num_total_patches,-1).unsqueeze(-1).unsqueeze(-1)                    # 60, 361, 1, 1
#     tensor = tensor.repeat(1,1,128,128)
#     tensor = tensor.reshape(row, col,kernel_size * kernel_size, 128, 128)
#     tensor = tensor.permute(0,3,1,4,2)                                    
#     tensor = tensor.reshape(768,1280, kernel_size * kernel_size)         # 768, 1280, 361

#     # kernels = torch.zeros((768,1280,kernel_size * kernel_size))
#     # kernels=kernels.unfold(0,128,128).unfold(1,128,128).reshape(60,kernel_size * kernel_size, 128, 128)

#     # kernels[:,:,:,:] = tensor
#     # kernels = kernels.reshape(row, col,kernel_size * kernel_size, 128, 128)  # 6, 10, 361, 128, 128


#     # kernels = kernels.permute(0, 3, 1, 4, 2)   # 6,128,10,128,361
#     # kernels=kernels.reshape(768, 1280, kernel_size * kernel_size)  # 768, 1280, 361

#     tensor = tensor[:orig_H,:orig_W,:]
#     tensor = tensor.permute(2,0,1)
    
#     return tensor