import numpy as np
# from castle.datasets import IIDSimulation
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from preprocess.preprocess_tab import TabularDataset
from torch.utils.data import DataLoader

from models.tabular_model import LinearRegression, MLPRegression

import networkx as nx
import matplotlib.pyplot as plt

from deeprobust.image.attack.pgd_mse import PGD


attack_params = {
    "PGD_reg": {
    'epsilon': 0.1,
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

d = 8
# causal_dag = np.array([[0., 1., 1., 0., 0., 0., 0., 0.],
#                        [0., 0., 0., 1., 0., 0., 0., 0.],
#                        [0., 0., 0., 1., 0., 0., 0., 0.],
#                        [0., 0., 0., 0., 0., 4., 1., 0.],
#                        [0., 0., 0., 0., 0., 1., 0., 0.],
#                        [0., 0., 0., 0., 0., 0., 1., 1.],
#                        [0., 0., 0., 0., 0., 0., 0., 1.],
#                        [0., 0., 0., 0., 0., 0., 0., 0.]])
# y_index = 3
# X = ut.simulate_linear_sem(causal_dag, 10000, 'gauss')
# np.savetxt('./data/tabular/syn_endo_{}.csv'.format(d), X, delimiter=',')


# d = 8
causal_dag = np.loadtxt('../../data/syn/syn_dag.csv', delimiter=',')
X = np.loadtxt('../../data/syn/syn_endo.csv', delimiter=',')
y_index = 3

label_dic = {}
variables_names = ['x(A)', 'x(P1)', 'x(P2)', 'y', 'x(CP)', 'x(C1)', 'x(C2)', 'x(D)']
for i in range(causal_dag.shape[0]):
    label_dic[i] = variables_names[i]

graph_nx = nx.from_numpy_array(causal_dag, create_using=nx.DiGraph)
nx.draw(graph_nx, labels=label_dic)
plt.show()

labels = X[:, y_index]
features = np.delete(X, y_index, axis=1)

print(min(labels), max(labels))

# # spiliting train and test
feat_train, feat_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, shuffle=True)
feat_train, feat_test, labels_train, labels_test = torch.FloatTensor(feat_train), \
                                                   torch.FloatTensor(feat_test), \
                                                   torch.FloatTensor(labels_train), \
                                                   torch.FloatTensor(labels_test)
dataset_train = TabularDataset(feat_train, labels_train)
dataset_test = TabularDataset(feat_test, labels_test)
dataloader_train = DataLoader(dataset_train, batch_size=1024, shuffle=True)
# dataloader_test = DataLoader(dataset_train, batch_size=1024, shuffle=True)


# training a PGD MLP model
mlp_reg_model_pgd = MLPRegression(features.shape[1], 32, 1)
optimizer = torch.optim.Adam(mlp_reg_model_pgd.parameters(), lr=0.0001)
mlp_reg_model_pgd.train()
for epoch in range(1500):
    for feat, label in dataloader_train:
        pgd = PGD(mlp_reg_model_pgd)
        feat_adv = pgd.generate(feat, label, **attack_params["PGD_reg"])
        preds_train_adv = mlp_reg_model_pgd.forward(feat_adv)
        loss_adv = mlp_reg_model_pgd.loss(preds_train_adv.squeeze(), label)
        optimizer.zero_grad()
        loss_adv.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss_adv))

# evaluate
mlp_reg_model_pgd.eval()
preds_test = mlp_reg_model_pgd.forward(feat_test)
mse_test = F.mse_loss(preds_test.squeeze(), labels_test, reduction='mean')

pgd = PGD(mlp_reg_model_pgd)
feat_test_pgd_adv = pgd.generate(feat_test, labels_test, **attack_params["PGD_reg"])
preds_test_pgd_adv = mlp_reg_model_pgd.forward(feat_test_pgd_adv)
mse_test_pgd_adv = F.mse_loss(preds_test_pgd_adv.squeeze(), labels_test, reduction='mean')

print("mse_test: {}, mse_adv: {}".format(mse_test, mse_test_pgd_adv))
# print(mlp_reg_model_pgd.state_dict())


# save checkpoinnt
cg_data = {
    'mse_test': mse_test
}
path = '../../ckpt/syn/syn_mlp_pgd_test.pth'
save_checkpoint(path, mlp_reg_model_pgd, optimizer, cg_data)


# training a PGD Linear model
lin_reg_model_pgd = LinearRegression(features.shape[1], 1)
optimizer = torch.optim.Adam(lin_reg_model_pgd.parameters(), lr=0.0001)
lin_reg_model_pgd.train()
for epoch in range(2500):
    for feat, label in dataloader_train:
        pgd = PGD(lin_reg_model_pgd)
        feat_adv = pgd.generate(feat, label, **attack_params["PGD_reg"])
        preds_train_adv = lin_reg_model_pgd.forward(feat_adv)
        loss_adv = lin_reg_model_pgd.loss(preds_train_adv.squeeze(), label)

        optimizer.zero_grad()
        loss_adv.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss_adv))

# evaluate
lin_reg_model_pgd.eval()
preds_test = lin_reg_model_pgd.forward(feat_test)
mse_test = F.mse_loss(preds_test.squeeze(), labels_test, reduction='mean')

pgd = PGD(lin_reg_model_pgd)
feat_test_pgd_adv = pgd.generate(feat_test, labels_test, **attack_params["PGD_reg"])
preds_test_pgd_adv = lin_reg_model_pgd.forward(feat_test_pgd_adv)
mse_test_pgd_adv = F.mse_loss(preds_test_pgd_adv.squeeze(), labels_test, reduction='mean')

print("mse_test: {}, mse_adv: {}".format(mse_test, mse_test_pgd_adv))


# save checkpoinnt
cg_data = {
    'mse_test': mse_test
}
path = '../../ckpt/syn/syn_lin_pgd_test.pth'
save_checkpoint(path, lin_reg_model_pgd, optimizer, cg_data)
