import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import numpy as np

# hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCH = 5

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self,):
        super().__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        logits = self.convolutional_layers(x)
        logits = self.flatten(logits)
        logits = self.linear_layers(logits)
        return logits


# save image to ./data/CIFAR10/sample.png
def save_img(img):
    # return normalization
    img = img / 2 + 0.5

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig('./data/CIFAR10/sample.png')


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function
    ) -> None:

    running_loss = 0.0

    for batch, (X,Y) in enumerate(dataloader):

        # feed forward
        predicted = model(X)
        # calculate loss
        loss = loss_function(predicted, Y)

        # reset gradients
        optimizer.zero_grad()
        # backward loss
        loss.backward()
        # update parameters
        optimizer.step()

        # display statistical information
        running_loss += loss.item()

        if batch % 1000 == 999:
            print('step   :', batch+1)
            print('loss   :', running_loss / 1000)
            running_loss = 0.0


def test(
    model: nn.Module,
    dataloader: DataLoader,
    ) -> None:

    correct = 0
    total = 0

    with torch.no_grad():
        for batch, (X,Y) in enumerate(dataloader):
            predicted = model(X)

            _, predicted = torch.max(predicted, 1)

            total += len(Y)
            correct += (predicted == Y).sum().item()

    print('accuracy :', 100 * correct/total)



def main():
    print('preparing data...')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = CIFAR10(
        root='./data/CIFAR10/',
        train=True,
        download=True,
        transform=transform,
    )

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    test_set = CIFAR10(
        root='./data/CIFAR10/',
        train=False,
        download=True,
        transform=transform,
    )

    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    classes = (
        'plane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
    )

    print('dataset has been successfully prepared!')

    # dataiter = iter(train_dataloader)
    # images, labels = dataiter.next()
    # save_img(torchvision.utils.make_grid(images))

    model = ConvolutionalNeuralNetwork()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM
    )

    print('data set size is', len(train_set))
    print('data loader size is', len(train_dataloader), '\n')

    print('train now...')
    for t in range(EPOCH):
        print('--- EPOCH', t+1 , '---')
        train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
        )
    print('done!')

    torch.save(model.state_dict(), './data/CIFAR10/model.pth')

    test(model=model, dataloader=test_dataloader)


if __name__=='__main__':
    main()