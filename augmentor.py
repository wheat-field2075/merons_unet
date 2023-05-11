#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torchvision import transforms as T


# In[ ]:


class Augmentor():
    def __init__(self, transforms=None):
        self.transforms = transforms
        
    def augment(self, iterable, rotate=False):
        s = np.random.randint(2147483647)
        
        record = []
        
        for i in iterable:
            np.random.seed(s)
            torch.manual_seed(s)
            if (self.transforms != None):
                i = self.transforms(i)
            if rotate:
                i = T.functional.rotate(i, int(np.random.choice([0, 90, 180, 270])))
            record.append(i)
        
        return tuple(record)

