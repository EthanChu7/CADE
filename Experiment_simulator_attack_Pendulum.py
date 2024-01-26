import os.path
import torch
import utils
import models.resnet as resnet
import models.vgg as vgg


import matplotlib.pyplot as plt
import os

from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.image as mpimg
import random
import math
import numpy as np
import pandas as pd 
if not os.path.exists('res_attack/sim_clip/'):
  os.makedirs('res_attack/sim_clip/')


def projection(theta, phi, x, y, base = -0.5):
    b = y-x*math.tan(phi)
    shade = (base - b)/math.tan(phi)
    return shade

def simulate_examples(attack_type='all', eps=0.1):
    scale = np.array([[0, 44], [100, 40], [7, 7.5], [10, 10]])
    count = 0
    empty = pd.DataFrame(columns=['i', 'j', 'shade', 'mid'])

    range_all = [[0., 50., 3., 3.], [49., 99., 13., 13.]]
    for i in range(0, 50):
        for j in range(50, 100):
            if j == 100:
                continue
            if count != 4:
                count += 1
                continue
            plt.rcParams['figure.figsize'] = (1.0, 1.0)
            theta = i*math.pi/200.0
            phi = j*math.pi/200.0

            x = 10 + 8 * math.sin(theta)
            y = 10.5 - 8*math.cos(theta)
            # print(x, y)

            if attack_type == '1' or attack_type == 'all':  # attack on light
                j_new = j + (np.random.rand() - 0.5) * 2 * eps * (range_all[1][1] - range_all[0][1])
                j_new = np.clip(j_new, range_all[0][1], range_all[1][1])
                phi = j_new * math.pi / 200.0

            ball = plt.Circle((x,y), 1.5, color = 'firebrick')
            gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)

            light = projection(theta, phi, 10, 10.5, 20.5)
            sun = plt.Circle((light,20.5), 3, color = 'orange')

            #calculate the mid index of
            ball_x = 10+9.5*math.sin(theta)
            ball_y = 10.5-9.5*math.cos(theta)
            mid = (projection(theta, phi, 10.0, 10.5)+projection(theta, phi, ball_x, ball_y))/2
            shade = max(3,abs(projection(theta, phi, 10.0, 10.5)-projection(theta, phi, ball_x, ball_y)))

            if attack_type == '2':  # shade_length:
                shade = shade + (np.random.rand() - 0.5) * 2 * eps * (range_all[1][2] - range_all[0][2])
                shade = np.clip(shade, range_all[0][2], range_all[1][2])
            elif attack_type == '3':  # shade_pos
                mid = mid + (np.random.rand() - 0.5) * 2 * eps * (range_all[1][3] - range_all[0][3])
                mid = np.clip(mid, range_all[0][3], range_all[1][3])
            elif attack_type == 'all':
                # j = j + (np.random.rand() - 0.5) * 2 * eps * (range_all[1][1] - range_all[0][1])
                shade = shade + (np.random.rand() - 0.5) * 2 * eps * (range_all[1][2] - range_all[0][2])
                mid = mid + (np.random.rand() - 0.5) * 2 * eps * (range_all[1][3] - range_all[0][3])
                shade = np.clip(shade, range_all[0][2], range_all[1][2])
                mid = np.clip(mid, range_all[0][3], range_all[1][3])

            shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)

            ax = plt.gca()
            ax.add_artist(gun)
            ax.add_artist(ball)
            ax.add_artist(sun)
            ax.add_artist(shadow)
            ax.set_xlim((0, 20))
            ax.set_ylim((-1, 21))
            new=pd.DataFrame({
                      'i':(i-scale[0][0])/(scale[0][1]-0),
                      'j':(j-scale[1][0])/(scale[1][1]-0),
                      'shade':(shade-scale[2][0])/(scale[2][1]-0),
                      'mid':(mid-scale[2][0])/(scale[2][1]-0)
                      },

                     index=[1])
            # empty=empty.append(new,ignore_index=True)
            plt.axis('off')

            if not os.path.exists('res_attack/sim_clip/type_{}_eps_{}'.format(attack_type, eps)):
                os.makedirs('res_attack/sim_clip/type_{}_eps_{}'.format(attack_type, eps))

            if count == 4:
              plt.savefig('res_attack/sim_clip/type_{}_eps_{}/a_'.format(attack_type, eps) + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(int(mid)) +'.png',dpi=96)
              count = 0

            plt.clf()
            count += 1

def pend_label_transform(label):
    # transform to 5 classes
    # label = (label // 10).long()
    label = (label).long()
    return label


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



device = 'cuda'
ckpt_resnet50 = torch.load("ckpt/pendulum/new_model_with_noise/pendulum_ResNet50_epoch30.pt", map_location=device)
ckpt_resnet50_pgd = torch.load('ckpt/pendulum/new_model_with_noise/pendulum_pgdtraining_8_resnet50_epoch80.pth', map_location=device)
ckpt_vgg16 = torch.load("ckpt/pendulum/new_model_with_noise/pendulum_vgg16_epoch120.pt", map_location=device)
ckpt_vgg16_pgd = torch.load('ckpt/pendulum/new_model_with_noise/pendulum_pgdtraining_8_vgg16_epoch80.pth', map_location=device)


num_classes = 50

model_resnet50 = resnet.ResNet50(num_classes=num_classes).to(device)
model_resnet50.load_state_dict(ckpt_resnet50)
model_resnet50_pgd = resnet.ResNet50(num_classes=num_classes).to(device)
model_resnet50_pgd.load_state_dict(ckpt_resnet50_pgd)
model_vgg16 = vgg.VGG('VGG16', num_classes=num_classes).to(device)
model_vgg16.load_state_dict(ckpt_vgg16)
model_vgg16_pgd = vgg.VGG('VGG16', num_classes=num_classes).to(device)
model_vgg16_pgd.load_state_dict(ckpt_vgg16_pgd)

model_resnet50.eval()
model_resnet50_pgd.eval()
model_vgg16.eval()
model_vgg16_pgd.eval()



target_label_idx = 0



epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
attack_types = ['1', '2', '3', 'all']
total_res = []
for attack_type in attack_types:
    l_asr_resnet50_base = []
    l_asr_resnet50_pgd_trans = []
    l_asr_vgg16_trans = []
    l_asr_vgg16_pgd_trans = []
    for eps in epsilons:
        print('simulating data...')
        simulate_examples(attack_type=attack_type, eps=eps)
        print('data simulated!')
        test_set = dataload_withlabel('res_attack/sim_clip/', image_size=64,
                                       mode='type_{}_eps_{}'.format(attack_type, eps))
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
        print('notice the size')
        size_test = len(test_loader.dataset)
        print(size_test)

        num_success_base = 0
        num_success_resnet50_pgd_trans = 0
        num_success_vgg16_trans = 0
        num_success_vgg16_pgd_trans = 0

        print("Attacking variables: ", attack_type)
        print("-------------------------------------------")

        for batch_id, (x, label) in enumerate(test_loader):
            label = label[:, target_label_idx]
            label = pend_label_transform(label)
            x = x.to(device)
            label = label.to(device)
            print(x.shape, label.shape)
            x_cra = x
            #
            # if not os.path.exists('res_attack/pendulum/cra_{}'.format(base_model)):
            #     os.makedirs('res_attack/pendulum/cra_{}'.format(base_model))

            pred_resnet50_base = torch.argmax(model_resnet50(x_cra), dim=1)
            pred_resnet50_pgd_trans = torch.argmax(model_resnet50_pgd(x_cra), dim=1)  # transfer to resnet50 pgd defense
            pred_vgg16_trans = torch.argmax(model_vgg16(x_cra), dim=1)  # transfer to vgg16
            pred_vgg16_pgd_trans = torch.argmax(model_vgg16_pgd(x_cra), dim=1)  # transfer to vgg16 pgd defense

            is_success_resnet50_base = (torch.abs(pred_resnet50_base - label) > 0).float()
            is_success_resnet50_pgd_trans = (torch.abs(pred_resnet50_pgd_trans - label) > 0).float()
            is_success_vgg16_trans = (torch.abs(pred_vgg16_trans - label) > 0).float()
            is_success_vgg16_pgd_trans = (torch.abs(pred_vgg16_pgd_trans - label) > 0).float()

            num_success_base_batch_i = torch.sum(is_success_resnet50_base)
            num_success_resnet50_pgd_trans_batch_i = torch.sum(is_success_resnet50_pgd_trans)
            num_success_vgg16_trans_batch_i = torch.sum(is_success_vgg16_trans)
            num_success_vgg16_pgd_trans_batch_i = torch.sum(is_success_vgg16_pgd_trans)

            print("batch_{} attacked!".format(batch_id))
            print("resnet50 asr: {}".format(num_success_base_batch_i / (label.shape[0])))
            print("transfer to resnet50 pgd defense, asr: {}".format(num_success_resnet50_pgd_trans_batch_i / (label.shape[0])))
            print("transfer to vgg16, asr: {}".format(num_success_vgg16_trans_batch_i / (label.shape[0])))
            print("transfer to vgg16 pgd defense, asr: {}".format(num_success_vgg16_pgd_trans_batch_i / (label.shape[0])))
            print("-------------------------------------------")

            num_success_base += num_success_base_batch_i
            num_success_resnet50_pgd_trans += num_success_resnet50_pgd_trans_batch_i
            num_success_vgg16_trans += num_success_vgg16_trans_batch_i
            num_success_vgg16_pgd_trans += num_success_vgg16_pgd_trans_batch_i

        asr_resnet50_base = num_success_base / size_test
        asr_resnet50_pgd_trans = num_success_resnet50_pgd_trans / size_test
        asr_vgg16_trans = num_success_vgg16_trans / size_test
        asr_vgg16_pgd_trans = num_success_vgg16_pgd_trans / size_test

        l_asr_resnet50_base.append(asr_resnet50_base.item())
        l_asr_resnet50_pgd_trans.append(asr_resnet50_pgd_trans.item())
        l_asr_vgg16_trans.append(asr_vgg16_trans.item())
        l_asr_vgg16_pgd_trans.append(asr_vgg16_pgd_trans.item())

        print("Attacking variables: ", attack_type)
        print("asr: {}".format(asr_resnet50_base))
        print("transfer to resnet50 pgd defense, asr: {}".format(asr_resnet50_pgd_trans))
        print("transfer to vgg16, asr: {}".format(asr_vgg16_trans))
        print("transfer to vgg16 pgd defense, asr: {}".format(asr_vgg16_pgd_trans))
        print("-------------------------------------------")

    print(l_asr_resnet50_base)
    print(l_asr_resnet50_pgd_trans)
    print(l_asr_vgg16_trans)
    print(l_asr_vgg16_pgd_trans)

    results = np.array([l_asr_resnet50_base,
                        l_asr_resnet50_pgd_trans,
                        l_asr_vgg16_trans,
                        l_asr_vgg16_pgd_trans])
    # results = results.transpose().flatten()
    print(results)
    total_res.append(results)

print(total_res)
for i in range(4):
    df = pd.DataFrame(total_res[i])
    df.to_csv('res_attack/sim/res_sim_eps_{}.csv'.format(attack_types[i]))





