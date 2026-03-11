'''
Standard Unet and attention Unet implementation

Author: Daniel Deidda

Copyright 2024 National Physical Laboratory.

SPDX-License-Identifier: Apache-2.0
'''

import torch
import torch.nn as nn
import math
 

class TwoC(nn.Module):
    """ double (convolution => [BN] => ReLU))"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.conv2(x)

class CLast(nn.Module):
    """out image using convolution and relu"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=False),
            #nn.PReLU()
        )
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upscaling using bilinear"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #self.bn = nn.BatchNorm2d(out_channels)
        #self.relu = nn.PReLU()

    def forward(self, x):
        x1 = self.up(x)
        return x1

class CDown(nn.Module):
    """Downsample (stride 2 convolution) => [BN] => lReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
    def forward(self, x):
        return self.down_conv(x)
    
class Attention_gate(nn.Module):
    """attention scheme"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=1,  padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.Ws = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, kernel_size=1,  padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=False)
        self.output=nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    #s for skip connection, g for the input
    def forward(self, g, s):
        Wg=self.Wg(g)
        Ws=self.Ws(s)
        reluout = self.relu(Wg+Ws)
        out = self.output(reluout)
        return out*s


class AttentionUp(nn.Module):
    """Upscaling using bilinear and attention"""

    def __init__(self, in_channels, out_channels):
        super(AttentionUp,self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.ag = Attention_gate(in_channels, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels[0]+out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x, s):
        up = self.up(x)
        ag = self.ag(up, s) 
        conc = torch.cat([up,s], axis=1)
        
        return self.conv(conc)
        
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
        #Encoder
        c1 = self.c2_1(inputs)
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

        return self.last(d1)
     
class AttentionUNet(nn.Module):
    def __init__(self, n_tot_in_channels, n_inter_channels, n_tot_out_channels):
        super(AttentionUNet, self).__init__()
        
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
        # self.up1 = AttentionUp( self.n_inter_channels*2*2*2,  self.n_inter_channels*2*2)
        # self.c2_5 = TwoC(self.n_inter_channels*2*2, self.n_inter_channels*2*2)
        # self.up2 = AttentionUp(self.n_inter_channels*2*2, self.n_inter_channels*2)
        # self.c2_6 = TwoC(self.n_inter_channels*2, self.n_inter_channels*2)
        # self.up3 = AttentionUp(self.n_inter_channels*2, self.n_inter_channels)
        # self.c2_7 = TwoC(self.n_inter_channels, self.n_inter_channels)
        # self.last = CLast(self.n_inter_channels, self.n_tot_out_channels)

        self.up1 = AttentionUp( [self.n_inter_channels*2*2*2,self.n_inter_channels*2*2],  self.n_inter_channels*2*2)
        self.c2_5 = TwoC(self.n_inter_channels*2*2, self.n_inter_channels*2*2)
        self.up2 = AttentionUp([self.n_inter_channels*2*2,self.n_inter_channels*2], self.n_inter_channels*2)
        self.c2_6 = TwoC(self.n_inter_channels*2, self.n_inter_channels*2)
        self.up3 = AttentionUp([self.n_inter_channels*2,self.n_inter_channels], self.n_inter_channels)
        self.c2_7 = TwoC(self.n_inter_channels, self.n_inter_channels)
        self.last = CLast(self.n_inter_channels, self.n_tot_out_channels)

    #dimension[batch, height, channels, width]
    def forward(self, inputs):
        #Encoder
        c1 = self.c2_1(inputs) #this is my skip
        c1d = self.d1(c1)
        c2 = self.c2_2(c1d)
        c2d = self.d2(c2)
        c3 = self.c2_3(c2d)
        c3d = self.d3(c3)


        c4 = self.c2_4(c3d)

        #Decoder
        d3u = self.up1(c4,c3) 
        d2u = self.up2(d3u,c2)
        d1u = self.up3(d2u,c1)

        return self.last(d1u)

    
