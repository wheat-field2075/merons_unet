#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
from PIL import Image
from tqdm import tqdm
from tensorboardX import SummaryWriter
import albumentations as A
import yaml


# In[2]:


sys.path.append('./modules')

from UNet import UNet
from Dataset import Dataset
from ImageLoader import ImageLoader


# In[3]:


data = yaml.load(open('./settings.yaml', 'r'), yaml.Loader)

images_path = data['images_path']
masks_path = data['masks_path']
image_patches_path = data['image_patches_path']
mask_patches_path = data['mask_patches_path']

patch_size = data['patch_size']
batch_size = data['batch_size']
sigma = data['sigma']
num_neg_samples = data['num_neg_samples']

folder = '2023.06.09 focal_loss_v2'
model_name = 'unet'


transform = A.Compose([
    A.RandomRotate90(p=1),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

train_set = []
val_set = []

for i in [10, 20, 30, 40]:
    train_set.append("Bubbles_movie_01_x1987x2020x81_3cv2_NLM_template20_search62_inverted{}.png".format(i))
for i in [50]:
    val_set.append("Bubbles_movie_01_x1987x2020x81_3cv2_NLM_template20_search62_inverted{}.png".format(i))


# In[4]:


train_ds = [[], []]
val_ds = [[], []]

for image_set, ds in [[train_set, train_ds], [val_set, val_ds]]:
    for image in image_set:
        print("Image:", image)
        patch_names = [file for file in os.listdir(os.path.join(image_patches_path, image)) if file[-4:] == '.npy']
        for patch in tqdm(range(len(patch_names))):
            image_patch = np.load(os.path.join(image_patches_path, image, patch_names[patch]))
            mask_patch = np.load(os.path.join(mask_patches_path, image, patch_names[patch]))
            
            ds[0].append(np.expand_dims(image_patch, 0))
            ds[1].append(np.expand_dims(mask_patch, 0))


# In[5]:


# input image: a batched numpy array or torch Tensor with dimensions (batch_size, 1, H, W)
# input mask: a batched numpy array or torch Tensor with dimensions (batch_size, 1, H, W)
# input transform: a transformation that is applied to each image and corresponding masks
# input scale_factor: the number of classes/intervals per mask
# output image: a batch numpy array with transformations applied and with dimensions (batch_size, 1, H, W)
# output mask_temp: a batch numpy array with transformations applied and with dimensions (batch_size, intervals, H, W)


def transform_data(image, mask, transform, scale_factor):
    if type(image) != np.array:
        image = np.array(image)
    if type(mask) != np.array:
        mask = np.array(mask)
    
    for batch in range(image.shape[0]):
        transformed = transform(image=np.moveaxis(image[batch], 0, -1), mask=np.moveaxis(mask[batch], 0, -1))
        image[batch] = np.moveaxis(transformed['image'], -1, 0)
        mask[batch] = np.moveaxis(transformed['mask'], -1, 0)
    
    mask_temp = np.zeros([mask.shape[0], intervals, mask.shape[2], mask.shape[3]])
    
    for interval in range(scale_factor):
        lower_bound = (1 / (scale_factor - 1)) * interval
        upper_bound = (1 / (scale_factor - 1)) * (interval + 1)
        mask_temp[:, interval] = ((mask >= lower_bound) * (mask < upper_bound)).squeeze()
        
    return image, mask_temp


# In[6]:


train_ds = Dataset(train_ds[0], train_ds[1])
val_ds = Dataset(val_ds[0], val_ds[1])

train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=batch_size)

epochs = 10000
lr = 5e-1
intervals = 9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unet = UNet(n_channels=1, n_classes=intervals).to(device)
unet = unet.to(device)
lossFunc = torchvision.ops.sigmoid_focal_loss
opt = torch.optim.SGD(unet.parameters(), lr=lr)

writer = SummaryWriter('./{}/runs/{}, lr={}'.format(folder, model_name, lr))


# In[7]:


for epoch in tqdm(range(epochs)):
    unet.train()
    total_train_loss = 0
    total_val_loss = 0

    for x, y in train_loader:
        x, y = transform_data(x, y, transform, intervals)
        x, y = torch.Tensor(x), torch.Tensor(y)

        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)
        
        pred = unet(x)
        loss = lossFunc(y, pred).mean()
        total_train_loss += loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        unet.eval()

        for x, y in val_loader:
            x, y = transform_data(x, y, transform, intervals)
            x, y = torch.Tensor(x), torch.Tensor(y)

            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            pred = unet(x)
            loss = lossFunc(y, pred).mean()
            total_val_loss += loss

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    
    writer.add_scalar('train_loss', avg_train_loss, epoch)
    writer.add_scalar('val_loss', avg_val_loss, epoch)
    
    if (epoch + 1) % 400 == 0:
        model_param_path = './{}/model_saves/{}, lr={}, epoch={}.pth'.format(folder, model_name, lr, epoch + 1)
        torch.save(unet.state_dict(), model_param_path)

writer.flush()
writer.close()

