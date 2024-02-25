from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self._len = features.shape[0]

    def __getitem__(self, item: int):
        return self.features[item], self.labels[item]

    def __len__(self):
        return self._len


def get_measurement_data(root='./data/', y_index=3):
    causal_dag = np.loadtxt(root + 'syn/syn_dag.csv', delimiter=',')
    endogenous = np.loadtxt(root + 'syn/syn_endo.csv', delimiter=',')

    # spiliting train and test
    endo_train, endo_test = train_test_split(endogenous, test_size=0.2, shuffle=True)
    # print("s", endo_train.shape)

    # split endogenous into features and labels
    feat_train = torch.tensor(np.delete(endo_train, y_index, axis=1), dtype=torch.float)
    feat_test = torch.tensor(np.delete(endo_test, y_index, axis=1), dtype=torch.float)
    labels_train = torch.tensor(endo_train[:, y_index], dtype=torch.float)
    labels_test = torch.tensor(endo_test[:, y_index], dtype=torch.float)
    endo_train = torch.tensor(endo_train, dtype=torch.float)
    endo_test = torch.tensor(endo_test, dtype=torch.float)
    causal_dag = torch.tensor(causal_dag, dtype=torch.float)

    print('endo_train:', endo_test[:5], endo_train.dtype)
    print('feat_train:', feat_test[:5], feat_train.dtype)
    print('labels_train:', labels_test[:5], labels_train.dtype)

    return feat_train, feat_test, labels_train, labels_test, endo_train, endo_test, causal_dag