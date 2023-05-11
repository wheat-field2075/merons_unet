#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


class ImageLoader():
    def __init__(self, img_path, mask_path, augs, image_size, images=None):
        self.img_path = img_path
        self.mask_path = mask_path
        if images == None:
            self.images = [file for file in os.listdir(self.img_path) if file[-4:] == '.png']
        else:
            self.images = images
        self.json_f = [file for file in os.listdir(self.img_path) if file[-5:] == '.json'][0]
        self.patches = []
        self.create_patches()
        self.augs = augs
        self.image_size = image_size
        
    def create_patches(self):
        for key in json.load(open(os.path.join(self.img_path, self.json_f)))['_via_img_metadata']:
            key = json.load(open(os.path.join(self.img_path, self.json_f)))['_via_img_metadata'][key]
            if key['filename'] not in self.images:
                continue
            for region in key['regions']:
                try:
                    if region['region_attributes']['Type'] != 'B':
                        continue
                except KeyError:
                    pass
                attr = region['shape_attributes']
                try:
                    self.patches.append(dict(fname=key['filename'], x=attr['x'], y=attr['y'], width=attr['width'], height=attr['height']))
                except KeyError:
                    pass
                
    def split(self, images):
        temp1 = ImageLoader(self.img_path, self.mask_path, self.augs, self.image_size, images=[image for image in self.images if image not in images])
        temp2 = ImageLoader(self.img_path, self.mask_path, self.augs, self.image_size, images=images)
        return temp1, temp2
    
    def __len__(self):
        return len(self.patches)
    
class PatchGenerator():
    def __init__(self, imgload):
        self.imgload = imgload
        self.image = None
        self.counter = 0
    def __iter__(self):
        return self
    def __next__(self):
        try:
            patch = self.imgload.patches[self.counter]
            if self.image == None or self.image['fname'] != patch['fname']:
                self.image = dict(fname=patch['fname'], image=plt.imread(os.path.join(self.imgload.img_path, patch['fname'])))
                self.mask = dict(fname=patch['fname'], mask=np.load(os.path.join(self.imgload.mask_path, patch['fname'] + '.npy')))
            self.counter += 1
            full_size = self.imgload.image_size 
            half_size = full_size // 2
            
            y = max(0, patch['y'] - half_size)
            x = max(0, patch['x'] - half_size)
            y = min(self.image['image'].shape[0] - full_size, y)
            x = min(self.image['image'].shape[1] - full_size, x)
            
            img_patch = self.image['image'][y:y+full_size, x:x+full_size]
            mask_patch = self.mask['mask'][y:y+full_size, x:x+full_size]
            
            if self.imgload.augs is not None:  
                img_patch, mask_patch = self.imgload.augs.augment([img_patch, mask_patch])

            return img_patch, mask_patch
            
        except IndexError:
            raise StopIteration

