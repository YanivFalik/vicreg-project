import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from hyperparams import batch_size

train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
test_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

to_tensor_transform = transforms.Compose([
    transforms.ToTensor()
])

class AugmentTwiceDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  
        return self.transform(img), self.transform(img), label

class PairedIndexDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, index_pairs, transform=None):
        self.base_dataset = base_dataset
        self.index_pairs = index_pairs
        self.transform = transform

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        i, j = self.index_pairs[idx]
        
        img1, label1 = self.base_dataset[i]
        img2, label2 = self.base_dataset[j]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, label1, img2, label2


def get_cifar_dataset():
    train_dataset = AugmentTwiceDataset(base_dataset=datasets.CIFAR10(root="./cifar_data", train=True, download=True), transform=train_transform)
    test_dataset = AugmentTwiceDataset(base_dataset=datasets.CIFAR10(root="./cifar_data", train=False, download=True), transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_cifar_dataset_test_transform():
    train_dataset = AugmentTwiceDataset(base_dataset=datasets.CIFAR10(root="./cifar_data", train=True, download=True), transform=test_transform)
    test_dataset = AugmentTwiceDataset(base_dataset=datasets.CIFAR10(root="./cifar_data", train=False, download=True), transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def raw_loader():
    # no transform needed for the pics
    train_dataset = datasets.CIFAR10(root="./cifar_data", train=True, download=True, transform=to_tensor_transform)
    test_dataset = datasets.CIFAR10(root="./cifar_data", train=False, download=True, transform=to_tensor_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader