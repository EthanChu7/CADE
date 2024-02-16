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
causal_dag = np.loadtxt('./data/syn/syn_dag.csv', delimiter=',')
X = np.loadtxt('./data/syn/syn_endo.csv', delimiter=',')
y_index = 3


label_dic = {}
for i in range(causal_dag.shape[0]):
    label_dic[i] = i + 1

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


# training a linear regression model
linear_reg_model = LinearRegression(features.shape[1], 1)
optimizer = torch.optim.Adam(linear_reg_model.parameters(), lr=1e-3)
linear_reg_model.train()

for epoch in range(1000):
    for feat, label in dataloader_train:
        optimizer.zero_grad()
        preds_train = linear_reg_model(feat)
        loss = linear_reg_model.loss(preds_train.squeeze(), label)
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss))

# evaluate
linear_reg_model.eval()
with torch.no_grad():
    preds_test = linear_reg_model(feat_test)
    mse_test = F.mse_loss(preds_test.squeeze(), labels_test, reduction='mean')
    print("mse_test: {}".format(mse_test))
    # print(linear_reg_model.state_dict())

# save checkpoinnt
cg_data = {
    'mse_test': mse_test
}
path = 'ckpt/syn/syn_lin_test.pth'
save_checkpoint(path, linear_reg_model, optimizer, cg_data)



mlp_reg_model = MLPRegression(features.shape[1], 32, 1)
optimizer = torch.optim.Adam(mlp_reg_model.parameters(), lr=1e-3)
mlp_reg_model.train()

for epoch in range(1000):
    for feat, label in dataloader_train:
        optimizer.zero_grad()
        preds_train = mlp_reg_model(feat)
        loss = mlp_reg_model.loss(preds_train.squeeze(), label)
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss))

# evaluate
mlp_reg_model.eval()
with torch.no_grad():
    preds_test = mlp_reg_model(feat_test)
    mse_test = F.mse_loss(preds_test.squeeze(), labels_test, reduction='mean')
    print("mse_test: {}".format(mse_test))
    # print(mlp_reg_model.state_dict())

# save checkpoinnt
cg_data = {
    'mse_test': mse_test
}
path = 'ckpt/syn/syn_mlp_test.pth'
save_checkpoint(path, mlp_reg_model, optimizer, cg_data)