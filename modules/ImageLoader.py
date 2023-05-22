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
