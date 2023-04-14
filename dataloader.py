import torchvision
from torch.utils.data import random_split
from torchvision import transforms


def generate_dataset():
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.FashionMNIST(root='', train=False, transform=transform, download=True)
    data_size = 5000  # choose 5000 images from the whole dataset
    other_data_size = 5000
    # choose 3500 images as train_set, choose 1000 images as val_set, choose 500 images as test_set,
    data_set, other_data_set = random_split(dataset, [data_size, other_data_size])
    train_size = int(data_size * 0.7)
    val_size = int(data_size * 0.2)
    test_size = int(data_size * 0.1)
    train_dataset, val_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset
