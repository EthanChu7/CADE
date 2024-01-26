from torch.utils.data.dataset import Dataset

class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self._len = features.shape[0]

    def __getitem__(self, item: int):
        return self.features[item], self.labels[item]

    def __len__(self):
        return self._len