import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    def __init__(self, dim_input, dim_out, bias=False):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(dim_input, dim_out, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out

    def loss(self, preds, labels):
        loss = F.mse_loss(preds, labels)
        return loss

class MLPRegression(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_out):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out

    def loss(self, preds, labels):
        loss = F.mse_loss(preds, labels)
        return loss