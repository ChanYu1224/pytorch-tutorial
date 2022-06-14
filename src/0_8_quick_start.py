import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import matplotlib.pyplot as plt


# gpu configuration
def gpu_configuration():
    if torch.cuda.is_available():
        print('cuda is available')
        chose = input('If you use GPU, select GPU number(0-3) :')
        if len(chose):
            DEVICE = 'cuda:' + chose
        else:
            DEVICE = 'cpu'
    else:
        DEVICE = 'cpu'

    print(DEVICE + ' is used!\n')

    return DEVICE


# initial values or hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCH = 5
DEVICE = gpu_configuration()


# configurate neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# train method
def train(dataloader, model, loss_function, optimizer, device):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device=device)
        y = y.to(device=device)

        # calculate error
        predicted = model(X)
        loss = loss_function(predicted, y)

        # reset gradient
        optimizer.zero_grad()
        # back propagation
        loss.backward()
        # adjast model values by calculated gradients
        optimizer.step()

        # display progress
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print('loss :', loss, '['+str(current)+'/'+str(size)+']')


# test method
def test(dataloader, model, loss_function, device):
    size = len(dataloader.dataset)
    
    # change to evaluation mode in model
    model.eval()
    
    # initial value
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device=device)
            y = y.to(device=device)

            # calculate error
            predicted = model(X)
            test_loss += loss_function(predicted, y).item()
            correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= size
        correct /= size
        print('---test error---')
        print('accuracy     :', 100*correct)
        print('average loss :', test_loss)


def main():
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

    print('downloaded successfully!\n')
    print('creating dataloader...')

    # create training dataloader
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    # create test dataloader
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    for X, y in test_dataloader:
        print('shape of X [N, C, H, W] :', X.shape)
        print('shape of y              :', y.shape)
        break

    print('dataloader created successfully!\n')
    print('creating model...')


    # set model
    model = NeuralNetwork().to(device=DEVICE)
    print('----model detail----')
    print(model)
    print('--------------------\n')

    # set loss function (cross entropy)
    loss_function = nn.CrossEntropyLoss()
    # set optimizer by SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # optimization
    for t in range(EPOCH):
        print('epoch', t+1)
        print('-----------------')
        train(dataloader=train_dataloader, model=model, loss_function=loss_function, optimizer=optimizer, device=DEVICE)
        test(dataloader=test_dataloader, model=model, loss_function=loss_function, device=DEVICE)
    print('Done!')


if __name__=='__main__':
    main()