import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import matplotlib.pyplot as plt


# initial values or hyperparameters
BATCH_SIZE = 64
GPU = [0]

# gpu configuration
if torch.cuda.is_available() and len(GPU):
    DEVICE = 'CUDA:'
    for gpu in GPU: DEVICE += str(gpu)
else:
    DEVICE = 'CPU'


print('now downloading...')

# download training data from datasets
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# download test data from datasets
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)

print('downloaded successfully!')
print('creating dataloader...')

# create training dataloader
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
# create test dataloader
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X, y in test_dataloader:
    print('shape of X [N, C, H, W] :', X.shape)
    print('shape of y              :', y.shape)
    break

print('dataloader created successfully!')
print('creating model...')


