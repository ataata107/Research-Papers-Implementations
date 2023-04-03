import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from torchvision.models import inception_v3
import os


class MRU_downsample(nn.Module):
    def __init__(self, in_channels, out_channels,concat_dim=3,downsample=True):
        super(MRU_downsample, self).__init__()
        self.conv_mi = nn.Conv2d(in_channels+concat_dim, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv_ni = nn.Conv2d(in_channels+concat_dim, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv_zi = nn.Conv2d(in_channels+concat_dim, in_channels, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.downsample = downsample

    def forward(self, x, img):
        mi = self.activation(self.conv_mi(torch.cat([x, img], dim=1))) 
        ni = self.activation(self.conv_ni(torch.cat([x, img], dim=1)))
        zi = self.activation(self.conv_zi(torch.cat([mi*x, img], dim=1)))
        y = (1 - ni) * zi + ni * x
        if(self.downsample==True) y = self.avg_pool(y)
        y = self.final_conv(y)
        return y

class MRU_upsample(nn.Module):
    def __init__(self, in_channels, out_channels,concat_dim=3,upsample_mode=True):
        super(MRU_upsample, self).__init__()
        self.conv_mi = nn.Conv2d(in_channels+concat_dim, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv_ni = nn.Conv2d(in_channels+concat_dim, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv_zi = nn.Conv2d(in_channels+concat_dim, in_channels, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.upsample_mode = upsample_mode

    def forward(self, x, img):
        mi = self.activation(self.conv_mi(torch.cat([x, img], dim=1)))
        ni = self.activation(self.conv_ni(torch.cat([x, img], dim=1)))
        zi = self.activation(self.conv_zi(torch.cat([mi*x, img], dim=1)))
        y = (1 - ni) * zi + ni * x
        # Upsample y by a factor of 2
        if(self.upsample_mode==True) y = self.upsample(y)
        y = self.final_conv(y)
        # Print the output dimensions
        print("Output shape:", y.shape)
        return y

class MRUGenerator(nn.Module): #3
    def __init__(self, in_channels=1, out_channels=3):
        super(MRUGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.initial_conv = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias = False)
        
        # Define 4 MRU_downsample layers
        self.down1 = MRU_downsample(1, 64,1)
        self.down2 = MRU_downsample(64, 128,1)
        self.down3 = MRU_downsample(128, 256,1)
        self.down4 = MRU_downsample(256, 512,1,False)
        
        # Define 4 MRU_upsample layers
        self.up1 = MRU_upsample(512, 256,1)
        self.up2 = MRU_upsample(256, 128,1)
        self.up3 = MRU_upsample(128, 64,1)
        self.up4 = MRU_upsample(64, self.out_channels,1,False)
        
    def forward(self, noise,img,img1,img2,img3):
        #img 1X3X64X64
        #x= self.initial_conv(img) : [1,64,64,64]
        # Downsample path
        x1 = self.down1(noise,img) # Output shape: [1, 64, 32, 32]
        x2 = self.down2(x1,img1) # Output shape: [1, 128, 16, 16]
        x3 = self.down3(x2,img2) # Output shape: [1, 256, 8, 8]
        x4 = self.down4(x3,img3) # Output shape: [1, 512, 8, 8]
        
        # Upsample path
        x5 = self.up1(x4, img3) # Output shape: [1, 256, 16, 16]
        x6 = self.up2(x5, img2) # Output shape: [1, 128, 32, 32]
        x7 = self.up3(x6, img1) # Output shape: [1, 64, 64, 64]
        y = self.up4(x7, img) # Output shape: [1, self.out_channels, 64, 64]
        
        return y


class MRU_Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_classes = 8):
        super(MRU_Discriminator, self).__init__()

        self.mru1 = nn.Sequential(
            MRU_downsample(input_channels, 64)
        )
        
        self.mru2 = nn.Sequential(
            MRU_downsample(64, 128)
        )
        self.mru2 = nn.Sequential(
            MRU_downsample(128, 256)
        )
        
        self.mru4 = nn.Sequential(
            MRU_downsample(256, 512,3,False)
        )
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=8, stride=1, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x,img,img1,img2,img3):
        # Downsample by a factor of 2
        x = self.mru1(x,img)
        # Downsample by a factor of 2
        x = self.mru2(x,img1)
        # Downsample by a factor of 2
        x = self.mru3(x,img2)
        # Downsample by a factor of 2
        x = self.mru4(x,img3)
        # Apply final layer and aux_layer
        x = self.final_layer(x)
        aux_classes = self.aux_layer(x)
        return x, aux_classes
