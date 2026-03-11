'''
Standard Unet and attention Unet implementation for 3D

Author: Daniel Deidda

Copyright 2024 National Physical Laboratory.

SPDX-License-Identifier: Apache-2.0
'''

import torch
import torch.nn as nn
import math
import numpy as np
 

class TwoC(nn.Module):
    """ double (convolution => [BN] => ReLU))"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

    def forward(self, x):
        return self.conv2(x)

class CLast(nn.Module):
    """out image using convolution and relu"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=False),
            #nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upscaling using bilinear"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )

    def forward(self, x):
        x1 = self.up(x)
        return x1

class CDown(nn.Module):
    """Downsample (stride 2 convolution) => [BN] => lReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )
    def forward(self, x):
        return self.down_conv(x)
    

def make_image_network_compatible(div,image):
    new_dim = []
    pad = []
    
    for dim in range(image.dim()):
        shape_it=image.shape[dim]
        
        if (shape_it % div!=0 and shape_it!=1 and dim>1):
            q = int(shape_it / div)
            new_dim.append((q+1)*div)
            if((new_dim[dim]-shape_it)%2==0):
                pad.append((int((new_dim[dim]-shape_it)/2),int((new_dim[dim]-shape_it)/2)))
            else:
                pad.append((int((new_dim[dim]-shape_it)/2),int((new_dim[dim]-shape_it)/2)+1))
        else:
            new_dim.append(image.shape[dim])
            pad.append((0,0))
    new_dim_tuple= tuple(new_dim)
    pad_tuple = tuple(pad)

    image_np=image.detach().cpu().numpy()
    image_np = np.pad(image_np,pad_tuple,'minimum')

    return torch.from_numpy(image_np)

def get_new_shape(new_shape, image):
    half_diff_dim = []
    has_negatives=False
    negatives_id = []
    for dim in range(image.dim()):
        #difference beween dimension
        half_diff_dim.append(int((image.shape[dim]-new_shape[dim])/2))
        if(half_diff_dim[dim]<0):
            has_negatives=True
            negatives_id.append(dim)


    if(not has_negatives):
        if(image.dim()==5):
            return image[half_diff_dim[0]:image.shape[0]-half_diff_dim[0], half_diff_dim[1]:image.shape[1]-half_diff_dim[1], 
                        half_diff_dim[2]:image.shape[2]-half_diff_dim[2], half_diff_dim[3]:image.shape[3]-half_diff_dim[3],
                        half_diff_dim[4]:image.shape[4]-half_diff_dim[4]]
    
    if(image.dim()==4):
        return image[half_diff_dim[0]:image.shape[0]-half_diff_dim[0], half_diff_dim[1]:image.shape[1]-half_diff_dim[1], 
                    half_diff_dim[2]:image.shape[2]-half_diff_dim[2], half_diff_dim[3]:image.shape[3]-half_diff_dim[3]]
    
    elif (negatives_id.__len__()==image.dim()):
        image_np=image.detach().cpu().numpy()
        return  np.pad(image_np,half_diff_dim,'minimum')
    else:
        print('ERROR: can only reshape a tensor to a tensor with all the dimension bigger or all smaller ex:'+
            '(1,2,3)->(1,3,4) or ->(1,1,2) I cannot do the following yet (1,2,3)->(1,3,2)')
       
         
class UNet(nn.Module):
    def __init__(self, n_tot_in_channels, n_inter_channels, n_tot_out_channels):
        super(UNet, self).__init__()
        
        self.n_tot_in_channels = n_tot_in_channels
        self.n_inter_channels = n_inter_channels
        self.n_tot_out_channels = n_tot_out_channels
        
        
        #Encoder
        self.c2_1 = TwoC(self.n_tot_in_channels, self.n_inter_channels)
        self.d1 = CDown(self.n_inter_channels, self.n_inter_channels)
        self.c2_2 = TwoC(self.n_inter_channels,self.n_inter_channels*2)
        self.d2 = CDown(self.n_inter_channels*2, self.n_inter_channels*2)
        self.c2_3 = TwoC(self.n_inter_channels*2, self.n_inter_channels*2*2)
        self.d3 = CDown(self.n_inter_channels*2*2, self.n_inter_channels*2*2)
        self.c2_4 = TwoC(self.n_inter_channels*2*2, self.n_inter_channels*2*2*2)

        #Decoder
        self.up1 = Up( self.n_inter_channels*2*2*2,  self.n_inter_channels*2*2)
        self.c2_5 = TwoC(self.n_inter_channels*2*2, self.n_inter_channels*2*2)
        self.up2 = Up(self.n_inter_channels*2*2, self.n_inter_channels*2)
        self.c2_6 = TwoC(self.n_inter_channels*2, self.n_inter_channels*2)
        self.up3 = Up(self.n_inter_channels*2, self.n_inter_channels)
        self.c2_7 = TwoC(self.n_inter_channels, self.n_inter_channels)
        self.last = CLast(self.n_inter_channels, self.n_tot_out_channels)
    
    #dimension[batch, height, channels, width]
    def forward(self, inputs):
        net_in = make_image_network_compatible(8, inputs)
        #Encoder
        c1 = self.c2_1(net_in)
        c1d = self.d1(c1)
        c2 = self.c2_2(c1d)
        c2d = self.d2(c2)
        c3 = self.c2_3(c2d)
        c3d = self.d3(c3)
        c4 = self.c2_4(c3d)

        #Decoder
        d3u = self.up1(c4)
        copy_add_3 = d3u+c3
        d3 = self.c2_5(copy_add_3)

        d2u = self.up2(d3)
        copy_add_2 = d2u+c2
        d2 = self.c2_6(copy_add_2)

        d1u = self.up3(d2)
        copy_add_1 = d1u+c1

        d1 = self.c2_7(copy_add_1)

        return get_new_shape(inputs.shape, self.last(d1))
    