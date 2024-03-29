import os.path
import torch
import models.resnet as resnet
import models.vgg as vgg
import matplotlib.pyplot as plt
import os
from preprocess.preprocess_img import dataload_withlabel
from torch.utils.data import DataLoader
import math
import numpy as np
import pandas as pd
from config_cade import get_config_pend


root_file_name = 'sim'

if not os.path.exists('res_attack/{}/'.format(root_file_name)):
  os.makedirs('res_attack/{}/'.format(root_file_name))


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

            plt.axis('off')

            if not os.path.exists('res_attack/{}/type_{}_eps_{}'.format(root_file_name, attack_type, eps)):
                os.makedirs('res_attack/{}/type_{}_eps_{}'.format(root_file_name, attack_type, eps))

            if count == 4:
              plt.savefig('res_attack/{}/type_{}_eps_{}/a_'.format(root_file_name, attack_type, eps) + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(int(mid)) +'.png',dpi=96)
              count = 0

            plt.clf()
            count += 1

def pend_label_transform(label):
    label = (label).long()
    return label


"""
LOAD DATASETS
"""

args = get_config_pend()
print(args)
device = 'cuda'

path_ckpt_resnet50 = args.path_ckpt_resnet50
path_ckpt_resnet50_pgd = args.path_ckpt_resnet50_pgd
path_ckpt_vgg16 = args.path_ckpt_vgg16
path_ckpt_vgg16_pgd = args.path_ckpt_vgg16_pgd

ckpt_resnet50 = torch.load(path_ckpt_resnet50, map_location=device)
ckpt_resnet50_pgd = torch.load(path_ckpt_resnet50_pgd, map_location=device)
ckpt_vgg16 = torch.load(path_ckpt_vgg16, map_location=device)
ckpt_vgg16_pgd = torch.load(path_ckpt_vgg16_pgd, map_location=device)

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
        test_set = dataload_withlabel('res_attack/{}/'.format(root_file_name), image_size=64,
                                       mode='type_{}_eps_{}'.format(attack_type, eps))
        np.random.seed(2)
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
    print(results)
    total_res.append(results)

print(total_res)
for i in range(4):
    df = pd.DataFrame(total_res[i])
    df.to_csv('res_attack/{}/res_sim_eps_{}.csv'.format(root_file_name, attack_types[i]))





