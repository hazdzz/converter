import torch
from torch.utils.data import Dataset


class DatasetCreator(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        X = self.data[index]
        Y = self.labels[index].to(dtype=torch.long)
        return (X, Y)

    def __len__(self):
        return len(self.labels)