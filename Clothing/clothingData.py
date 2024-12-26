import numpy as np
import torch
import torch.utils
from PIL import Image
import os
from tqdm import tqdm
import torchvision.transforms as transforms


class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, img_transform):
        self.root = root
        flist = os.path.join(root, "annotations/clean_label_kv.txt")
        self.flist_noise = os.path.join(root, "annotations/noisy_label_kv.txt")
        self.imlist = self.flist_reader(flist)
        self.transform = img_transform

    def __getitem__(self, index):
        impath, target_clean, target_noise = self.imlist[index]
        img = Image.open(impath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target_clean, target_noise

    def __len__(self):
        return len(self.imlist)

    def flist_reader(self, flist):
        imlist = []
        not_found = 0
        count = 0

        noise_dict = {}
        with open(self.flist_noise, 'r') as noise_file:
            for noise_line in noise_file:
                noise_path, noise_label = noise_line.strip().split(" ")
                noise_dict[noise_path] = int(noise_label)
        
        with open(flist, 'r') as rf:
            for line in tqdm(rf):
                row = line.strip().split(" ")
                impath = self.root + row[0]
                imlabel = int(row[1])

                imlabel_noise = noise_dict.get(row[0])

                if imlabel_noise is not None:
                    imlist.append((impath, imlabel, imlabel_noise))
                else:
                    imlist.append((impath, imlabel, imlabel))
                    not_found += 1
                
                count += 1

        print("Not found:", not_found)
        return imlist



transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
         ])