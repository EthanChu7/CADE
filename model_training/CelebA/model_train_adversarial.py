from deeprobust.image.defense.pgdtraining import PGDtraining
from deeprobust.image.attack.pgd import PGD
import torch
from torchvision import datasets, transforms
import deeprobust.image.netmodels.resnet as resnet


defense_params = {
    "PGDtraining_CelebA": {
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name': "celeba_pgdtraining.pt",
        'epsilon': 8.0 / 255.0,
        'clip_max': 1.0,
        'clip_min': -1.0,
        'epoch_num': 80,
        'num_steps': 10,
        'lr': 1e-2,
    },
}

"""
LOAD DATASETS
"""

trans_f = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = datasets.CelebA('data', split='train', download=False, transform=trans_f)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=64,
                                           shuffle=True,
                                           pin_memory=False,
                                           drop_last=True, num_workers=0)

test_set = datasets.CelebA('data', split='test', download=False, transform=trans_f)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=32,
                                          shuffle=True,
                                          pin_memory=False,
                                          drop_last=False, num_workers=0)


"""
TRAIN DEFENSE MODEL
"""

num_classes = 2

print('====== START TRAINING =====')
device = 'cuda'
model = resnet.ResNet50(num_classes=num_classes).to(device)

defense = PGDtraining(model, device)
defense.generate(train_loader, test_loader, label_idx=39, **defense_params["PGDtraining_CelebA"])

print('====== FINISH TRAINING =====')

