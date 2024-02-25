import numpy as np
import torch
import torch.nn.functional as F
from preprocess.preprocess_tab import TabularDataset, get_measurement_data
from torch.utils.data import DataLoader
from models.tabular_model import LinearRegression, MLPRegression
from deeprobust.image.attack.pgd_mse import PGD


attack_params = {
    "PGD_reg": {
    'epsilon': 0.1, # [0.1, 0.2, 0.3, 0.4, 0.5]
    'clip_max': None,
    'clip_min': None,
    'print_process': False,
    'num_steps': 5,
    'bound': 'linf'
    }
}

def save_checkpoint(path, model, optimizer, cg_data):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cg_data": cg_data
    }, path)


np.random.seed(0)
torch.manual_seed(0)

y_index = 3
feat_train, feat_test, labels_train, labels_test, endo_train, endo_test, causal_dag = get_measurement_data(root='../../data/', y_index=y_index)


dataset_train = TabularDataset(feat_train, labels_train)
dataset_test = TabularDataset(feat_test, labels_test)
dataloader_train = DataLoader(dataset_train, batch_size=1024, shuffle=True)
# dataloader_test = DataLoader(dataset_train, batch_size=1024, shuffle=True)


# training a PGD MLP model
mlp_reg_model_pgd = MLPRegression(feat_train.shape[1], 32, 1)
optimizer = torch.optim.Adam(mlp_reg_model_pgd.parameters(), lr=0.0001)
mlp_reg_model_pgd.train()
for epoch in range(1500):
    for feat, label in dataloader_train:
        optimizer.zero_grad()
        pgd = PGD(mlp_reg_model_pgd)
        feat_adv = pgd.generate(feat, label, **attack_params["PGD_reg"])
        preds_train_adv = mlp_reg_model_pgd(feat_adv)
        loss_adv = mlp_reg_model_pgd.loss(preds_train_adv.squeeze(), label)
        loss_adv.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss_adv))

# evaluate
mlp_reg_model_pgd.eval()
with torch.no_grad():
    preds_test = mlp_reg_model_pgd(feat_test)
    mse_test = F.mse_loss(preds_test.squeeze(), labels_test, reduction='mean')

pgd = PGD(mlp_reg_model_pgd)
feat_test_pgd_adv = pgd.generate(feat_test, labels_test, **attack_params["PGD_reg"])
with torch.no_grad():
    preds_test_pgd_adv = mlp_reg_model_pgd(feat_test_pgd_adv)
    mse_test_pgd_adv = F.mse_loss(preds_test_pgd_adv.squeeze(), labels_test, reduction='mean')

print("mse_test: {}, mse_adv: {}".format(mse_test, mse_test_pgd_adv))
# print(mlp_reg_model_pgd.state_dict())


# save checkpoinnt
cg_data = {
    'mse_test': mse_test
}
path = '../../ckpt/syn/syn_mlp_pgd.pth'
save_checkpoint(path, mlp_reg_model_pgd, optimizer, cg_data)


# training a PGD Linear model
lin_reg_model_pgd = LinearRegression(feat_train.shape[1], 1)
optimizer = torch.optim.Adam(lin_reg_model_pgd.parameters(), lr=0.0001)
lin_reg_model_pgd.train()
for epoch in range(2500):
    for feat, label in dataloader_train:
        optimizer.zero_grad()
        pgd = PGD(lin_reg_model_pgd)
        feat_adv = pgd.generate(feat, label, **attack_params["PGD_reg"])
        preds_train_adv = lin_reg_model_pgd(feat_adv)
        loss_adv = lin_reg_model_pgd.loss(preds_train_adv.squeeze(), label)

        loss_adv.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss_adv))

# evaluate
lin_reg_model_pgd.eval()
with torch.no_grad():
    preds_test = lin_reg_model_pgd(feat_test)
    mse_test = F.mse_loss(preds_test.squeeze(), labels_test, reduction='mean')

pgd = PGD(lin_reg_model_pgd)
feat_test_pgd_adv = pgd.generate(feat_test, labels_test, **attack_params["PGD_reg"])
with torch.no_grad():
    preds_test_pgd_adv = lin_reg_model_pgd(feat_test_pgd_adv)
    mse_test_pgd_adv = F.mse_loss(preds_test_pgd_adv.squeeze(), labels_test, reduction='mean')

print("mse_test: {}, mse_adv: {}".format(mse_test, mse_test_pgd_adv))


# save checkpoinnt
cg_data = {
    'mse_test': mse_test
}
path = '../../ckpt/syn/syn_lin_pgd.pth'
save_checkpoint(path, lin_reg_model_pgd, optimizer, cg_data)
