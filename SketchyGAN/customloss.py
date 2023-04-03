import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import os

#Define Focal Loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Define the perceptual loss function
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.inception = inception_v3(pretrained=True, aux_logits=False)
        self.inception.eval()
        self.layers = nn.ModuleList([self.inception.Conv2d_2a_3x3,
                                     self.inception.Mixed_3a,
                                     self.inception.Mixed_4a,
                                     self.inception.Mixed_5a])
        for param in self.inception.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        y = nn.functional.interpolate(y, size=(299, 299), mode='bilinear', align_corners=False)
        for l in self.layers:
            x_features = l(x)
            y_features = l(y)
            loss += torch.mean(torch.abs(x_features - y_features))
        return loss
