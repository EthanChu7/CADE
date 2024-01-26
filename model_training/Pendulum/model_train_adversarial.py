from deeprobust.image.defense.pgdtraining_pendulum import PGDtraining
from deeprobust.image.attack.pgd import PGD
import torch
from torchvision import datasets, transforms
import deeprobust.image.netmodels.resnet as resnet
import deeprobust.image.netmodels.vgg as vgg
# from deeprobust.image.config import defense_params
from torch.utils.data import TensorDataset, DataLoader

import os
import pandas as pd
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

defense_params = {
    "PGDtraining_pendulum": {
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name': "pendulum_pgdtraining",
        'epsilon': 8.0 / 255.0,
        'epoch_num': 200,
        'clip_max': 1.0,
        'clip_min': 0.,
        'num_steps': 10,
        'lr': 1e-4,
    }
}


class dataload_withlabel(torch.utils.data.Dataset):
    def __init__(self, root, label_file=None, image_size=64, mode="train", sup_prop=1., num_sample=0):
        # label_file: 'pendulum_label_downstream.txt'

        self.label_file = label_file
        if label_file is not None:
            self.attrs_df = pd.read_csv(os.path.join(root, label_file))
            # attr = self.attrs_df[:, [1,2,3,7,5]]
            self.split_df = pd.read_csv(os.path.join(root, label_file))
            splits = self.split_df['partition'].values
            split_map = {
                "train": 0,
                "valid": 1,
                "test": 2,
                "all": None,
            }
            split = split_map[verify_str_arg(mode.lower(), "split",
                                             ("train", "valid", "test", "all"))]
            mask = slice(None) if split is None else (splits == split)
            self.mask = mask
            np.random.seed(2)
            if num_sample > 0:
                idxs = [i for i, x in enumerate(mask) if x]
                not_sample = np.random.permutation(idxs)[num_sample:]
                mask[not_sample] = False
            self.attrs_df = self.attrs_df.values
            self.attrs_df[self.attrs_df == -1] = 0
            self.attrs_df = self.attrs_df[mask][:, [0,1,2,3,6]]
            self.imglabel = torch.as_tensor(self.attrs_df.astype(np.float))
            self.imgs = []
            for i in range(3):
                mode1 = list(split_map.keys())[i]
                root1 = root + mode1
                imgs = os.listdir(root1)
                self.imgs += [os.path.join(root, mode1, k) for k in imgs]
            self.imgs = np.array(self.imgs)[mask]
        else:
            root = root + mode
            imgs = os.listdir(root)
            self.imgs = [os.path.join(root, k) for k in imgs]
            self.imglabel = [list(map(float, k[:-4].split("_")[1:])) for k in imgs]
        self.transforms = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor()])
        np.random.seed(2)
        self.n = len(self.imgs)
        self.available_label_index = np.random.choice(self.n, int(self.n * sup_prop), replace=0)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        if not (idx in self.available_label_index):
            label = torch.zeros(4).long() - 1
        else:
            if self.label_file is None:
                label = torch.from_numpy(np.asarray(self.imglabel[idx]))
            else:
                label = self.imglabel[idx]
        pil_img = Image.open(img_path).convert('RGB')
        array = np.array(pil_img)
        array1 = np.array(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,3)
            data = torch.from_numpy(pil_img)
        return data, label.float()

    def __len__(self):
        return len(self.imgs)


"""
LOAD DATASETS
"""
train_set = dataload_withlabel('data/pendulum/', image_size=64,
                               mode='train')
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True, num_workers=0)

test_set = dataload_withlabel('data/pendulum/', image_size=64,
                              mode='test')
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=0)


"""
TRAIN DEFENSE MODEL
"""

num_classes = 50  # 0 - 49

print('====== START TRAINING =====')
device = 'cuda'
# model = resnet.ResNet50(num_classes=num_classes).to(device)
model = vgg.VGG('VGG16', num_classes=num_classes).to(device)


defense = PGDtraining(model, device)
defense.generate(train_loader, test_loader, label_idx=0, **defense_params["PGDtraining_pendulum"])

print('====== FINISH TRAINING =====')

