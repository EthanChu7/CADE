"""
This function help to train model of different archtecture easily. Select model archtecture and training data, then output corresponding model.

"""
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

from torch.utils.data import TensorDataset, DataLoader

def train(model, data, device, maxepoch, data_path = './', save_per_epoch = 10, seed = 100, image_size=64, lr=0.1):
    """train.

    Parameters
    ----------
    model :
        model(option:'CNN', 'ResNet18', 'ResNet34', 'ResNet50', 'densenet', 'vgg11', 'vgg13', 'vgg16', 'vgg19')
    data :
        data(option:'MNIST','CIFAR10')
    device :
        device(option:'cpu', 'cuda')
    maxepoch :
        training epoch
    data_path :
        data path(default = './')
    save_per_epoch :
        save_per_epoch(default = 10)
    seed :
        seed

    Examples
    --------
    >>>import deeprobust.image.netmodels.train_model as trainmodel
    >>>trainmodel.train('CNN', 'MNIST', 'cuda', 20)
    """

    torch.manual_seed(seed)

    if data == 'CelebA':
        num_classes = 2
    elif data == 'pendulum':
        num_classes = 50

    print("num_classes: ", num_classes)

    train_loader, test_loader = feed_dataset(data, data_path)

    if (model == 'CNN'):
        import deeprobust.image.netmodels.CNN as MODEL
        #from deeprobust.image.netmodels.CNN import Net
        train_net = MODEL.Net(num_classes=num_classes).to(device)

    elif (model == 'ResNet18'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet18(num_classes=num_classes).to(device)

    elif (model == 'ResNet34'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet34(num_classes=num_classes).to(device)

    elif (model == 'ResNet50'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet50(num_classes=num_classes).to(device)

    elif (model == 'ResNet50_pretrained'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet50_pre_trained(num_classes=num_classes).to(device)

    elif (model == 'densenet'):
        import deeprobust.image.netmodels.densenet as MODEL
        train_net = MODEL.densenet_cifar().to(device)

    elif (model == 'vgg11'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG11', num_classes=num_classes).to(device)
    elif (model == 'vgg13'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG13', num_classes=num_classes).to(device)
    elif (model == 'vgg16'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG16', num_classes=num_classes).to(device)
    elif (model == 'vgg19'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG19', num_classes=num_classes).to(device)
    elif (model == 'vitb16'):
        import deeprobust.image.netmodels.vit as MODEL
        train_net = MODEL.vitb16(image_size, num_classes).to(device)
    elif (model == 'vitb32'):
        import deeprobust.image.netmodels.vit as MODEL
        train_net = MODEL.vitb32(image_size, num_classes).to(device)
    elif (model == 'vitl16'):
        import deeprobust.image.netmodels.vit as MODEL
        train_net = MODEL.vitl16(image_size, num_classes).to(device)
    elif (model == 'vitl32'):
        import deeprobust.image.netmodels.vit as MODEL
        train_net = MODEL.vitl32(image_size, num_classes).to(device)



    optimizer = optim.Adam(train_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
    save_model = True
    for epoch in range(1, maxepoch + 1):     ## 5 batches
        print(epoch)
        if data == 'CelebA':
            MODEL.train_celeba(train_net, device, train_loader, optimizer, epoch, label_idx=39)
            MODEL.test_celeba(train_net, device, test_loader, label_idx=39)
        elif data == 'pendulum':
            MODEL.train_pend(train_net, device, train_loader, optimizer, epoch, label_idx=0)
            MODEL.test_pend(train_net, device, test_loader, label_idx=0)

        if (save_model and (epoch % (save_per_epoch) == 0 or epoch == maxepoch)):
            if os.path.isdir('./trained_models/'):
                print('Save model.')
                torch.save(train_net.state_dict(), os.path.join('trained_models', data + "_" + model + "_epoch" + str(epoch) + ".pt"))
            else:
                os.mkdir('./trained_models/')
                print('Make directory and save model.')
                torch.save(train_net.state_dict(), os.path.join('trained_models', data + "_" + model + "_epoch_" + str(epoch) + ".pt"))
        scheduler.step()

def feed_dataset(data, data_dict):
    if(data == 'CIFAR10'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=True, download = True,
                        transform=transform_train),
                 batch_size= 128, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=False, download = True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True) #, **kwargs)

    elif(data == 'MNIST'):
        train_loader = torch.utils.data.DataLoader(
                 datasets.MNIST(data_dict, train=True, download = True,
                 transform=transforms.Compose([transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])),
                 batch_size=128,
                 shuffle=True)

        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(data_dict, train=False, download = True,
                transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=1000,
                shuffle=True)

    elif(data == 'ImageNet'):
        pass

    elif(data == 'CelebA'):
        trans_f = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CelebA('data', split='train', download=False, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=1024,
                                                   shuffle=True,
                                                   pin_memory=False,
                                                   drop_last=True, num_workers=0)

        test_set = datasets.CelebA('data', split='test', download=False, transform=trans_f)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                   batch_size=1024,
                                                   shuffle=True,
                                                   pin_memory=False,
                                                   drop_last=False, num_workers=0)

    elif data == 'pendulum':
        train_set = dataload_withlabel('data/pendulum/', image_size=64,
                                       mode='train')
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, num_workers=0)

        test_set = dataload_withlabel('data/pendulum/', image_size=64,
                                       mode='test')
        test_loader = DataLoader(test_set, batch_size=128, shuffle=True, drop_last=False, num_workers=0)

        # _, label = next(iter(train_loader))
        # print(label)

    return train_loader, test_loader


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









