from functools import reduce

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import USPS
from torchvision import transforms


def get_train_test_dataloaders(root, dataset, batch_size=32, num_workers=4, drop_last=False):
    if dataset == 'USPS':
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        train_dataset = USPS(root=root, train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  num_workers=num_workers, shuffle=True, 
                                  drop_last=drop_last)
        
        test_dataset = USPS(root=root, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=drop_last)
        
    else:
        raise ValueError(dataset)
        
    return train_loader, test_loader