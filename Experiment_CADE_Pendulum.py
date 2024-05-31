import os.path
import pandas as pd
import numpy as np
import torch
from generative_model.bgm import BGM
from config_cade import get_config_pend
from preprocess.preprocess_img import make_dataloader
from attacker_cade import CADELatent
from torchvision.utils import save_image

import models.resnet as resnet
import models.vgg as vgg


def pend_label_transform(label):
    label = (label).long()
    return label


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic=True

    # load args and ckpts
    args = get_config_pend()
    print(args)
    device = torch.device('cuda')

    substitute = args.substitute
    epsilon = args.epsilon
    type_loss = args.type_loss
    num_steps = args.num_steps
    lr = args.lr
    path_ckpt_generative = args.path_ckpt_generative
    path_ckpt_resnet50 = args.path_ckpt_resnet50
    path_ckpt_resnet50_pgd = args.path_ckpt_resnet50_pgd
    path_ckpt_vgg16 = args.path_ckpt_vgg16
    path_ckpt_vgg16_pgd = args.path_ckpt_vgg16_pgd

    ckpt_generative = torch.load(path_ckpt_generative, map_location=device)  # load the ckpt of the pretrained causal generative model
    ckpt_resnet50 = torch.load(path_ckpt_resnet50, map_location=device)
    ckpt_resnet50_pgd = torch.load(path_ckpt_resnet50_pgd, map_location=device)
    ckpt_vgg16 = torch.load(path_ckpt_vgg16, map_location=device)
    ckpt_vgg16_pgd = torch.load(path_ckpt_vgg16_pgd, map_location=device)

    # causal graph for Pendulum
    A = torch.zeros((4, 4))
    A[0, 2:4] = 1
    A[1, 2:4] = 1
    num_classes = 50
    target_label_idx = 0  # the index of label pendulum_angle


    print('Build models...')
    model = BGM(latent_dim=6,
                conv_dim=32,
                image_size=64,
                enc_dist='gaussian',
                enc_arch='resnet',
                enc_fc_size=1024,
                enc_noise_dim=128,
                dec_dist='implicit',
                prior='nlrscm',
                num_label=4,
                A=A)
    model.load_state_dict(ckpt_generative['model'])
    model.eval()
    model.to(device)

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

    print('Model loaded')



    # load dataset
    data_loader = make_dataloader(args)
    size_dataset = len(data_loader.dataset)
    print("The size of the dataset:", size_dataset)

    l_attacking_nodes = [[1], [2], [3], [1, 2, 3]] # z1 (light_angle), z2 (shadow_length), z3 (shadow_position), z1:3
    l_causal = [True, False, False, False]  # since variables z_2 and z_3 both have no descendants in z, thus setting False to avoid extra computations
    ranges = [[6.8932], [13.4833], [6.9335], [6.8932, 13.4833, 6.9335]]  # the range of the latent z

    if substitute == 'resnet50':
        model_white = model_resnet50
    elif substitute == 'vgg16':
        model_white = model_vgg16
    model_white.eval()

    l_asr_resnet50 = []
    l_asr_resnet50_pgd = []
    l_asr_vgg16 = []
    l_asr_vgg16_pgd = []

    for mode in range(len(l_attacking_nodes)):
        attacker = CADELatent(model, attacking_nodes=l_attacking_nodes[mode], substitute=model_white, device=device)
        num_success_resnet50 = 0
        num_success_resnet50_pgd = 0
        num_success_vgg16 = 0
        num_success_vgg16_pgd = 0

        print("Attacking variables: ", l_attacking_nodes[mode])
        print("-------------------------------------------")

        for batch_id, (x, label) in enumerate(data_loader):
            label = label[:, target_label_idx]
            label = pend_label_transform(label)
            x = x.to(device)
            label = label.to(device)

            x_cade = attacker.attack_whitebox(x, label, lr=lr, epochs=num_steps, type_loss=type_loss, epsilon=epsilon * torch.tensor(ranges[mode], dtype=torch.float).to(device), causal_layer=l_causal[mode])
            x_cade = x_cade.clamp(0, 1)

            if not os.path.exists('res_attack/pendulum/cade_{}'.format(substitute)):
                os.makedirs('res_attack/pendulum/cade_{}'.format(substitute))
            np.save('res_attack/pendulum/cade_{}/x_cade_batch_{}.npy'.format(substitute, batch_id), x_cade.detach().cpu().numpy())

            with torch.no_grad():
                pred_resnet50 = torch.argmax(model_resnet50(x_cade), dim=1)  # transfer to resnet50
                pred_resnet50_pgd = torch.argmax(model_resnet50_pgd(x_cade), dim=1)  # transfer to resnet50 pgd defense
                pred_vgg16 = torch.argmax(model_vgg16(x_cade), dim=1)  # transfer to vgg16
                pred_vgg16_pgd = torch.argmax(model_vgg16_pgd(x_cade), dim=1)  # transfer to vgg16 pgd defense

            is_success_resnet50 = (pred_resnet50 != label).float()
            is_success_resnet50_pgd = (pred_resnet50_pgd != label).float()
            is_success_vgg16 = (pred_vgg16 != label).float()
            is_success_vgg16_pgd = (pred_vgg16_pgd != label).float()

            num_success_resnet50_batch_i = torch.sum(is_success_resnet50)  # number of success attacks in batch i
            num_success_resnet50_pgd_batch_i = torch.sum(is_success_resnet50_pgd)
            num_success_vgg16_batch_i = torch.sum(is_success_vgg16)
            num_success_vgg16_pgd_batch_i = torch.sum(is_success_vgg16_pgd)

            print("batch_{} attacked!".format(batch_id))
            print("resnet50, asr: {}".format(num_success_resnet50_batch_i / (label.shape[0])))
            print("resnet50 pgd defense, asr: {}".format(num_success_resnet50_pgd_batch_i / (label.shape[0])))
            print("vgg16, asr: {}".format(num_success_vgg16_batch_i / (label.shape[0])))
            print("vgg16 pgd defense, asr: {}".format(num_success_vgg16_pgd_batch_i / (label.shape[0])))
            print("-------------------------------------------")

            num_success_resnet50 += num_success_resnet50_batch_i
            num_success_resnet50_pgd += num_success_resnet50_pgd_batch_i
            num_success_vgg16 += num_success_vgg16_batch_i
            num_success_vgg16_pgd += num_success_vgg16_pgd_batch_i

            if not os.path.exists('res_attack/pendulum/cade_{}/batch_{}'.format(substitute, batch_id)):
                os.makedirs('res_attack/pendulum/cade_{}/batch_{}'.format(substitute, batch_id))

            for x_id in range((x_cade.shape[0])):
                x_ori_i = x[x_id]
                x_cade_i = x_cade[x_id]

                is_success_resnet50_i = is_success_resnet50[x_id]
                is_success_resnet50_pgd_i = is_success_resnet50_pgd[x_id]
                is_success_vgg16_i = is_success_vgg16[x_id]
                is_success_vgg16_pgd_i = is_success_vgg16_pgd[x_id]

                save_image(x_ori_i.detach().cpu(), "res_attack/pendulum/cade_{}/batch_{}/x_{}_ori.png".format(substitute, batch_id, x_id), normalize=True, scale_each=True)
                save_image(x_cade_i.detach().cpu(), "res_attack/pendulum/cade_{}/batch_{}/x_{}_cade_mode({})_{}_{}_{}_{}.png".format(substitute, batch_id, x_id, mode, int(is_success_resnet50_i), int(is_success_resnet50_pgd_i), int(is_success_vgg16_i), int(is_success_vgg16_pgd_i)), normalize=True, scale_each=True)

        asr_resnet50_base = num_success_resnet50 / size_dataset
        asr_resnet50_pgd_trans = num_success_resnet50_pgd / size_dataset
        asr_vgg16_trans = num_success_vgg16 / size_dataset
        asr_vgg16_pgd_trans = num_success_vgg16_pgd / size_dataset

        l_asr_resnet50.append(asr_resnet50_base.item())
        l_asr_resnet50_pgd.append(asr_resnet50_pgd_trans.item())
        l_asr_vgg16.append(asr_vgg16_trans.item())
        l_asr_vgg16_pgd.append(asr_vgg16_pgd_trans.item())

        print("Attacking variables: ", l_attacking_nodes[mode])
        print("resnet50 asr: {}".format(asr_resnet50_base))
        print("resnet50 pgd asr: {}".format(asr_resnet50_pgd_trans))
        print("vgg16 asr: {}".format(asr_vgg16_trans))
        print("vgg16 pgd asr: {}".format(asr_vgg16_pgd_trans))
        print("-------------------------------------------")

    print(l_asr_resnet50)
    print(l_asr_resnet50_pgd)
    print(l_asr_vgg16)
    print(l_asr_vgg16_pgd)

    results = np.array([l_asr_resnet50,
                        l_asr_resnet50_pgd,
                        l_asr_vgg16,
                        l_asr_vgg16_pgd])
    results = results.transpose()
    print(results)
    df = pd.DataFrame(results, columns=['ResNet50', 'ResNet50 (defense)', 'VGG16', 'VGG16 (defense)'])
    df.to_csv('res_attack/pendulum/cade_{}/res_{}_base_cade.csv'.format(substitute, substitute))


if __name__ == '__main__':
    main()