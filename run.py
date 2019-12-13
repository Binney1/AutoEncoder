import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import ConvAE

# Transforms
TRANSFORM_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

TRANSFORM_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# Settings
settings = {
    'transform': TRANSFORM_MNIST,
    'train_batch_size': 32,
    'test_batch_size': 4,
    'num_workers': 1,
}

dataset =  torchvision.datasets.MNIST

# Load the data
trainset = dataset(root='./datasets', train=True, download=True, transform=settings['transform'])
testset = dataset(root='./datasets', train=False, download=True, transform=settings['transform'])

train_dataloader = utils.DataLoader(trainset, batch_size=settings['train_batch_size'], shuffle=True, num_workers=settings['num_workers'])
test_dataloader = utils.DataLoader(testset, batch_size=settings['test_batch_size'], shuffle=True, num_workers=settings['num_workers'])

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
