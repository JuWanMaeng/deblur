
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import torch
import torchvision as v
matplotlib.use('PS')


# spatially variant blur
class BatchBlur_SV(nn.Module):
    def __init__(self, l=19, padmode='reflection'):
        super(BatchBlur_SV, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))


    def forward(self, input, kernel):
        # input shape 1,3,256,256  kernel shape 1,361,256,256
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            pad = pad.view(C * B, 1, H_p, W_p)
            pad = F.unfold(pad, self.l).transpose(1, 2)         # torch.Size([3 * B, 65536, 361])
            kernel = kernel.flatten(2).unsqueeze(0)
            kernel_sum = torch.sum(kernel, axis=(0,1,2), keepdim=True)
            kernel = kernel / kernel_sum
            kernel = kernel.expand(3, -1, -1, -1)              # torch.Size([3, B, 65536, 361])

            # out_unf = (pad*kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))).sum(2).unsqueeze(1)   # origin [3B, 65536,361]
            out_unf = (pad*kernel.contiguous().view(-1, kernel.size(3), kernel.size(2))).sum(2).unsqueeze(1)  # mine
            out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)
            
            return out


class BatchBlur_SV_NAF(nn.Module):
    def __init__(self, l=19, padmode='reflection'):
        super(BatchBlur_SV_NAF, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))


    def forward(self, input, kernel):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        pad = self.pad(input)         # 10, 3, 274, 274
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))   # 30, 3, 274, 274
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            b_img = torch.zeros_like(input)
            for i in range(3):
                data = pad[:,i:i+1,:,:]
                data = data.view(1 * B, 1, H_p, W_p)            # 30, 1, 274, 274  
                data = F.unfold(data, self.l).transpose(1, 2)
                data = data.permute(0,2,1).reshape(1,H*W,1,self.l,self.l).squeeze(0)
                data = data.permute(1,0,2,3)
    
                kernel = kernel.permute(2,3,0,1)  # 256,256,1,361
                kernel = kernel.contiguous().view(-1,1,19,19)
                kernel_sum = torch.sum(kernel,axis=(1,2,3), keepdim=True)
                kernel = kernel / kernel_sum


                output = F.conv2d(data, kernel, stride=1, groups=kernel.shape[0])
                output = output.contiguous().view(output.shape[0],output.shape[1],-1) 
                output = output.permute(0,2,1)
                out_unf = F.fold(output, output_size=(256,256), kernel_size=1, stride=1)

                

            return out_unf