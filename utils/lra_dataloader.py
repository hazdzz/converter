import torch

class DatasetCreator(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        X = self.data[index]
        Y = self.labels[index].to(dtype=torch.long)
        
        return (X, Y)

    def __len__(self):
        return len(self.labels)


def count_params(net):
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    return n_params