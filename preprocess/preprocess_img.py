import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import os

import pandas as pd


def draw_recon(x, x_recon):
    x_l, x_recon_l = x.tolist(), x_recon.tolist()
    result = [None] * (len(x_l) + len(x_recon_l))
    result[::2] = x_l
    result[1::2] = x_recon_l
    return torch.FloatTensor(result)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def make_dataloader(args):

    data_loader = None
    if args.dataset == 'celeba':

        trans_f = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # print(args.data_dir)
        dataset = datasets.CelebA(args.data_dir, download=False, transform=trans_f)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False,
                                                   drop_last=False)

    elif 'pendulum' == args.dataset:
        dataset = dataload_withlabel(args.data_dir, image_size = args.image_size,
                                       mode='test')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return data_loader


def check_for_CUDA(sagan_obj):
    if not sagan_obj.config.disable_cuda and torch.cuda.is_available():
        print("CUDA is available!")
        sagan_obj.device = torch.device('cuda')
        sagan_obj.config.dataloader_args['pin_memory'] = True
    else:
        print("Cuda is NOT available, running on CPU.")
        sagan_obj.device = torch.device('cpu')

    if torch.cuda.is_available() and sagan_obj.config.disable_cuda:
        print("WARNING: You have a CUDA device, so you should probably run without --disable_cuda")


class dataload_withlabel(torch.utils.data.Dataset):
    def __init__(self, root, image_size=64, mode="train"):
        # label_file: 'pendulum_label_downstream.txt'

        root = root + mode
        imgs = os.listdir(root)
        imgs.sort()
        # print(imgs)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.imglabel = [list(map(float, k[:-4].split("_")[1:])) for k in imgs]

        # print(np.min(self.imglabel, axis=0))
        # print(np.max(self.imglabel, axis=0))
        #
        # print('the final modified version')
        # print("mean:", np.mean(self.imglabel, axis=0))
        # print("std", np.std(self.imglabel, axis=0))

        self.transforms = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = torch.from_numpy(np.asarray(self.imglabel[idx]))

        pil_img = Image.open(img_path).convert('RGB')

        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,3)
            data = torch.from_numpy(pil_img)
        return data, label.float()

    def __len__(self):
        return len(self.imgs)
