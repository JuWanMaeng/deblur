
# # 영역 크기 (64, 64)
# patch_size = (64, 64)

# # 스트라이드 (stride) 크기 (64, 64)
# stride = (64, 64)


# tensor = kernel
# tensor = tensor.unfold(1, patch_size[0], stride[0]).unfold(2,patch_size[1],stride[1]).reshape(1,-1,patch_size[0],patch_size[1]).permute(1,0,2,3)
# tensor = F.interpolate(tensor, size=(19, 19), mode='bilinear', align_corners=False)
# tensor = tensor.view(60,-1).unsqueeze(-1).unsqueeze(-1)
# # tensor = ((tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)

# kernels = torch.zeros((768,1280,361))
# kernels=kernels.unfold(0,128,128).unfold(1,128,128).reshape(60, 19 * 19, 128, 128)

# kernels[:,:,:,:] = tensor
# kernels = kernels.reshape(6,10,361,128,128)

# kernels = kernels.permute(0,3,1,4,2)
# kernels=kernels.reshape(768,1280,361)

# kernel = kernels[:720,:1280,:]