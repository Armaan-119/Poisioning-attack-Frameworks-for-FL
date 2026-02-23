from torchvision import datasets, transforms
from torch.utils.data import random_split

def load_datasets(num_clients):

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients

    return random_split(dataset, lengths)
