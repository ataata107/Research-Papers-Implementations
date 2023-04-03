import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from torchvision.models import inception_v3
from customdataset import CustomDataset
from mru import MRUGenerator, MRU_Discriminator
from customloss import FocalLoss, PerceptualLoss
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes=8

# Define hyperparameters
lr = 0.0001
batch_size = 8
epochs = 50
img_size = 64
train_img_paths = "./data"

# define custom transform to resize images and sketches

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# define train and test dataset using the custom dataset class and transform

dataset = CustomDataset(train_img_paths, transform=transform)

# define data loader to return both image and their corresponding sketches

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define generator and discriminator instances

generator = MRUGenerator().to(device)
discriminator = MRU_Discriminator().to(device)


# Define the Inception-V3 model
inception = inception_v3(pretrained=True, aux_logits=False)
inception.eval()

# define loss function and optimizer for generator and discriminator

# Define loss function and optimizer
criterion = FocalLoss()#torch.nn.BCELoss()
aux_criterion = torch.nn.CrossEntropyLoss()
l1_loss = torch.nn.L1Loss()
perc_loss = PerceptualLoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr*2, betas=(0.5, 0.999))

# train the model
fixed_noise = torch.randn(batch_size, 1, 64, 64).to(device)
for epoch in range(epochs):
    for batch_idx, (img, sketch,real_label_class) in enumerate(train_loader):
        img = img.to(device)
        # resize image and sketch for generator,discriminator
        img1 = torch.nn.functional.interpolate(img, scale_factor=0.5, mode='bilinear').to(device)
        img2 = torch.nn.functional.interpolate(img, scale_factor=0.25, mode='bilinear').to(device)
        img3 = torch.nn.functional.interpolate(img, scale_factor=0.125, mode='bilinear').to(device)

        sketch = sketch.to(device)
        sktch1 = torch.nn.functional.interpolate(sketch, scale_factor=0.5, mode='bilinear').to(device)
        sktch2 = torch.nn.functional.interpolate(sketch, scale_factor=0.25, mode='bilinear').to(device)
        sktch3 = torch.nn.functional.interpolate(sketch, scale_factor=0.125, mode='bilinear').to(device)
        
        # set generator and discriminator gradients to zero
        generator.zero_grad()
        discriminator.zero_grad()
        
        # generate fake image from sketch using generator
        fake_img = generator(fixed_noise, sketch, sktch1, sktch2, sktch3)
        fake_img1 = torch.nn.functional.interpolate(fake_img.detach(), scale_factor=0.5, mode='bilinear').to(device)
        fake_img2 = torch.nn.functional.interpolate(fake_img.detach(), scale_factor=0.5, mode='bilinear').to(device)
        fake_img3 = torch.nn.functional.interpolate(fake_img.detach(), scale_factor=0.5, mode='bilinear').to(device)
        
        # train discriminator with real and fake images
        real_pred, real_aux_classes = discriminator(img, img1, img2, img3)
        fake_pred, fake_aux_classes = discriminator(fake_img.detach(), fake_img1, fake_img2, fake_img3)
        disc_loss_1 = (criterion(real_pred, torch.ones_like(real_pred)) + 
                     criterion(fake_pred, torch.zeros_like(fake_pred))) / 2
        disc_loss_2 = aux_criterion(fake_aux_classes, real_label_class)
        disc_loss = disc_loss_1+disc_loss_2
        
        # backpropagate loss and update discriminator weights
        disc_loss.backward()
        optimizer_D.step()
        
        # Train generator
        generator.zero_grad()
        fake_output,fake_aux_classes = discriminator(fake_img.detach(), fake_img1, fake_img2, fake_img3)
        real_labels = torch.ones(fake_output.size(), device=device)
        g_loss_1 = criterion(fake_output, real_labels)
        g_loss_2 = aux_criterion(fake_aux_classes, real_label_class)
        g_loss_3 = l1_loss(fake_img, img)
        g_loss_4 = perc_loss(fake_img, img)
        g_loss = g_loss_1 + g_loss_2 + g_loss_3  + g_loss_4
        g_loss.backward()
        optimizer_G.step()
