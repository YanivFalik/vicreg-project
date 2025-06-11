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

        return img1, img2, label1

class IndexedCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label, index

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
    train_dataset = IndexedCIFAR10(root="./cifar_data", train=True, download=True, transform=test_transform)
    test_dataset = IndexedCIFAR10(root="./cifar_data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_pairwise_dataloader(pairs):
    base_dataset = datasets.CIFAR10(root="./cifar_data", train=True, download=True, transform=None)
    paired_dataset = PairedIndexDataset(base_dataset=base_dataset, index_pairs=pairs, transform=test_transform)
    paired_dataloader = DataLoader(paired_dataset, batch_size, shuffle=True)
    return paired_dataloader

def get_one_img_per_class(transform=test_transform, num_classes=10):
    base_dataset = datasets.CIFAR10(root="./cifar_data", train=True, download=True)

    class_to_idx = {}
    selected_imgs = []

    for i in range(len(base_dataset)):
        img, label = base_dataset[i]
        if label not in class_to_idx:
            class_to_idx[label] = i
            transformed_img = transform(img)
            selected_imgs.append((transformed_img, label, i)) 
            if len(class_to_idx) == num_classes:
                break

    return selected_imgs

def get_base_dataset():
    return datasets.CIFAR10(root="./cifar_data", train=True, download=True)


# Anomaly Detection part
# we have to transform mnist images [1, 28, 28] to [3, 32, 32] so they can fit the encoder
mnist_shape_to_cifar_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),  # to get 3 channels from 1
    transforms.ToTensor(),
])

class ADDataset(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset, mnist_dataset):
        self.cifar = cifar_dataset
        self.mnist = mnist_dataset
        self.cifar_len = len(self.cifar)
    
    def __len__(self):
        return len(self.cifar) + len(self.mnist)
    
    def __getitem__(self, idx):
        if (idx < self.cifar_len):
            img, _ = self.cifar[idx]
            # cifar images are labeld 0 = not anomolous
            return img, 0 
        else: 
            img, _ = self.mnist[idx - self.cifar_len]
            # mnist images are labeled 1 = anomolous
            return img, 1

def get_ad_train_and_test_dataloader():
    cifar_test_dataset = datasets.CIFAR10(root="./cifar_data", train=False, download=True, transform=test_transform)
    mnist_test_dataset = datasets.MNIST(root="./mnist_data", train=False, download=True, transform=mnist_shape_to_cifar_transform)
    test_loader = DataLoader(dataset=ADDataset(cifar_dataset=cifar_test_dataset, mnist_dataset=mnist_test_dataset), batch_size=batch_size, shuffle=False)

    train_loader, _ = raw_loader()
    return train_loader, test_loader