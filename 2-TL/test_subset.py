# This is just a test to see how Subset datasets work in PyTorch.

import torch
from torch.utils.data import Dataset, Subset, DataLoader


class myDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(0, 10)
        self.train_indices = [0,1,2,3,4,5,6]
        self.test_indices = [7,8,9]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    
    
if __name__ == '__main__':
    dataset = myDataset()
    train_dataset = Subset(dataset, dataset.train_indices)
    test_dataset = Subset(dataset, dataset.test_indices)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print(train_loader.dataset.dataset.train_indices)