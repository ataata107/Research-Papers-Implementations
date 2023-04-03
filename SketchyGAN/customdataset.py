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


#Dataloader class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None,num_classes=8):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._get_samples()
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, sketch_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            sketch = self.transform(sketch)
        
        return img, sketch, torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes)
    
    def _get_samples(self):
        samples = []
        label_folders = sorted(os.listdir(self.root_dir))
        for label in label_folders:
            if not os.path.isdir(os.path.join(self.root_dir, label)):
                continue
            img_dir = os.path.join(self.root_dir, label, 'img')
            sketch_dir = os.path.join(self.root_dir, label, 'sketch')
            if not os.path.isdir(img_dir) or not os.path.isdir(sketch_dir):
                continue
            img_filenames = sorted(os.listdir(img_dir))
            sketch_filenames = sorted(os.listdir(sketch_dir))
            for i in range(len(img_filenames)):
                img_path = os.path.join(img_dir, img_filenames[i])
                sketch_path = os.path.join(sketch_dir, sketch_filenames[i])
                samples.append((img_path, sketch_path, int(label)))
        return samples
