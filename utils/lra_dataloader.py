import torch
from torch.utils.data import Dataset


class SingleDatasetCreator(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        datum = self.data[idx]
        label = self.labels[idx].to(dtype=torch.long)
        
        return (datum, label)
    

class DualDatasetCreator(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        datum1 = self.data1[idx]
        datum2 = self.data2[idx]
        label = self.labels[idx].to(dtype=torch.long)

        return (datum1, datum2, label)


def count_params(net):
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    return n_params