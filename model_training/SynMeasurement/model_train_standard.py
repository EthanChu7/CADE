import numpy as np
import torch
import torch.nn.functional as F
from preprocess.preprocess_tab import TabularDataset, get_measurement_data
from torch.utils.data import DataLoader
from models.tabular_model import LinearRegression, MLPRegression



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


# training a linear regression model
linear_reg_model = LinearRegression(feat_train.shape[1], 1)
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
    print(linear_reg_model.state_dict())

# save checkpoinnt
cg_data = {
    'mse_test': mse_test
}
path = '../../ckpt/syn/syn_lin.pth'
save_checkpoint(path, linear_reg_model, optimizer, cg_data)



mlp_reg_model = MLPRegression(feat_train.shape[1], 32, 1)
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
path = '../../ckpt/syn/syn_mlp.pth'
save_checkpoint(path, mlp_reg_model, optimizer, cg_data)