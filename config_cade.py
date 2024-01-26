import argparse


def get_config_pend():

    parser = argparse.ArgumentParser(description='CADE_pend')

    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='pendulum', choices=['celeba', 'pendulum'])
    parser.add_argument('--substitute', type=str, default='resnet50', choices=['resnet50', 'vgg16'])
    parser.add_argument('--data_dir', type=str, default='data/pendulum/', help='data directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--step_size', type=float, default=0.4)
    parser.add_argument('--type_loss', type=str, default='pendulum')
    parser.add_argument('--path_ckpt_generative', type=str, default='ckpt/pendulum/pendulum_bgm.pt', help='ckpt path')
    parser.add_argument('--path_ckpt_resnet50', type=str, default='ckpt/pendulum/pendulum_resnet50.pt', help='ckpt path')
    parser.add_argument('--path_ckpt_resnet50_pgd', type=str, default='ckpt/pendulum/pendulum_defense_resnet50_pgd.pt', help='ckpt path')
    parser.add_argument('--path_ckpt_vgg16', type=str, default='ckpt/pendulum/pendulum_vgg16.pt', help='ckpt path')
    parser.add_argument('--path_ckpt_vgg16_pgd', type=str, default='ckpt/pendulum/pendulum_defense_vgg16_pgd.pt', help='ckpt path')

    args = parser.parse_args()

    return args


def get_config_celeba():

    parser = argparse.ArgumentParser(description='CADE_celeba')

    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='celeba', choices=['celeba', 'pendulum'])
    parser.add_argument('--substitute', type=str, default='resnet50', choices=['resnet50', 'vgg16'])
    parser.add_argument('--data_dir', type=str, default='data/', help='data directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=200)
    parser.add_argument('--epsilon', type=float, default=0.7)
    parser.add_argument('--step_size', type=float, default=0.5)
    parser.add_argument('--type_loss', type=str, default='celeba')
    parser.add_argument('--path_ckpt_generative', type=str, default='ckpt/celeba/celeba_bgm.pt', help='ckpt path')
    parser.add_argument('--path_ckpt_resnet50', type=str, default='ckpt/celeba/celeba_resnet50.pt', help='ckpt path')
    parser.add_argument('--path_ckpt_resnet50_pgd', type=str, default='ckpt/celeba/celeba_defense_resnet50_pgd.pt', help='ckpt path')
    parser.add_argument('--path_ckpt_vgg16', type=str, default='ckpt/celeba/celeba_vgg16.pt', help='ckpt path')
    parser.add_argument('--path_ckpt_vgg16_pgd', type=str, default='ckpt/celeba/celeba_defense_vgg16_pgd.pt', help='ckpt path')

    args = parser.parse_args()

    return args


def get_config_syn():

    parser = argparse.ArgumentParser(description='CADE_syn')

    parser.add_argument('--substitute', type=str, default='lin', choices=['lin', 'mlp'])
    parser.add_argument('--num_steps', type=int, default=150)
    parser.add_argument('--epsilons', type=list, default=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--path_lin', type=str, default='ckpt/syn/syn_lin.pth', help='ckpt path')
    parser.add_argument('--path_lin_pgd', type=str, default='ckpt/syn/syn_lin_pgd.pth', help='ckpt path')
    parser.add_argument('--path_mlp', type=str, default='ckpt/syn/syn_mlp.pth', help='ckpt path')
    parser.add_argument('--path_mlp_pgd', type=str, default='ckpt/syn/syn_mlp_pgd.pth', help='ckpt path')

    args = parser.parse_args()

    return args