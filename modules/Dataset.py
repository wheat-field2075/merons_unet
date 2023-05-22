import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks):
        self.imgs = imgs
        self.masks = masks

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        im = self.imgs[index]
        mask = self.masks[index]
        
        # format to H, W, C
        im = im.reshape([im.shape[0], im.shape[1], 1])
        mask = mask.reshape([mask.shape[0], mask.shape[1], 1])
                                    
        return tuple([im, mask])