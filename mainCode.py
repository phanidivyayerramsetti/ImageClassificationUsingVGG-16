import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import random_split
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = .40),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_training_dataloader(train_transform, batch_size = 128, num_workers= 0, shuffle = True):
    transform_train = train_transform
    cifar10_training = torchvision.datasets.CIFAR10(root = '.', train = True, download = True, transform = transform_train)
    cifar10_training_loader = DataLoader(cifar10_training, shuffle = shuffle, num_workers = num_workers, batch_size = batch_size)

    return cifar10_training_loader

def get_testing_dataloader(test_transform, batch_size = 128, num_workers = 0, shuffle = True):
    transform_test = test_transform
    cifar10_test = torchvision.datasets.CIFAR10(root = '.', train = False, download = True, transform = transform_test)
    cifar10_test_loader = DataLoader(cifar10_test, shuffle = shuffle, num_workers = num_workers, batch_size = batch_size)

    return cifar10_test_loader

trainloader = get_training_dataloader(train_transform)
testloader = get_testing_dataloader(test_transform)
classes = {0 : 'airplane', 1 : 'automobile', 2 : 'bird', 3 : 'cat', 4 : 'deer', 5 : 'dog', 6 : 'frog', 7 : 'horse', 8 : 'ship', 9 : 'truck'}
fig, ax = plt.subplots(5, 5, figsize = (10, 10))

for batch_idx, (inputs, labels) in enumerate(trainloader):
    for im in range(25):
        image = inputs[im].permute(1, 2, 0)
        i = im // 5
        j = im % 5
        ax[i, j].imshow(image.numpy())
        ax[i, j].axis('off')
        ax[i, j].set_title(classes[int(labels[im].numpy())])

    break;

plt.suptitle('CIFAR-10 Images')
plt.show()

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels, inner_channel, kernel_size = 1, bias = False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size = 3, padding = 1, bias = False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias = False),
            nn.AvgPool2d(2, stride = 2)
        )

    def forward(self, x):
        return self.down_sample(x)


class VGG16(nn.Module):
    def __init__(self, block, nblocks, growth_rate = 12, reduction = 0.5, num_class = 10):
        super().__init__()
        self.growth_rate = growth_rate

        inner_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size = 3, padding = 1, bias = False)
        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module('dense_block_layer_{}'.format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            out_channels = int(reduction * inner_channels)

            self.features.add_module('transition_layer_{}'.format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module('dense_block{}'.format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks) - 1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('activation', nn.ReLU(inplace = True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def VGG2(activation = 'relu'):
    return VGG16(Bottleneck, [6, 12, 24, 16], growth_rate = 32)

epochs = 5
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

model = VGG2()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = learning_rate)
train_stats = pd.DataFrame(columns = ['Epoch', 'Time per epoch', 'Avg time per step', 'Train loss',
                                     'Train Accuracy', 'Train Top-3 Accuracy', 'Test Loss', 'Test Accuracy', 'Test Top-3 Accuracy'])
model.to(device)

steps = 0
running_loss = 0
for epoch in range(epochs):

    since = time.time()

    train_accuracy = 0
    top3_train_accuracy = 0
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        np_top3_class = ps.topk(3, dim = 1)[1].cpu().numpy()
        target_numpy = labels.cpu().numpy()
        top3_train_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

    time_elapsed = time.time() - since

    test_loss = 0
    test_accuracy = 0
    top3_test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            np_top3_class = ps.topk(3, dim = 1)[1].cpu().numpy()
            target_numpy = labels.cpu().numpy()
            top3_test_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

        print(f'Epoch {epoch+1} / {epochs}.. '
             f'Time per epoch : {time_elapsed:.4f}..'
             f'Average time per step : {time_elapsed/len(trainloader):.4f}..'
             f'Train loss : {running_loss/len(trainloader):.4f}..'
             f'Train Accuracy : {train_accuracy/len(trainloader):.4f}..'
             f'Top-3 train Accuracy : {top3_train_accuracy/len(trainloader):.4f}..'
             f'Test loss : {test_loss/len(testloader):.4f}..'
             f'Test Accuracy : {test_accuracy/len(testloader):.4f}..'
             f'Top-3 test Accuracy : {top3_test_accuracy/len(testloader):.4f}..')

        train_stats = train_stats.append({'Epoch' : epoch, 'Time per epoch' : time_elapsed, 'Avg time per step' : time_elapsed/len(trainloader),
                                         'Train loss' : running_loss/len(trainloader), 'Train accuracy' : train_accuracy/len(trainloader),
                                         'Train top-3 accuracy' : top3_train_accuracy/len(trainloader), 'Test loss' : test_loss/len(testloader),
                                         'Test accuracy' : test_accuracy/len(testloader), 'Test top-3 accuracy' : top3_test_accuracy/len(testloader)}, ignore_index = True)

        running_loss = 0
        model.train()

        fig = plt.figure(figsize = (10, 7))
ax = plt.axes()

plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

x = range(1, len(train_stats['Train loss'].values) + 1)
ax.plot(x, train_stats['Train loss'].values, '-g', label = 'train loss');
ax.plot(x, train_stats['Test loss'].values, '-b', label = 'test loss');

plt.legend()

fig = plt.figure(figsize = (10, 7))
ax = plt.axes()

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

x = range(1, len(train_stats['Train accuracy'].values) + 1)
ax.plot(x, train_stats['Train accuracy'].values, '-g', label = 'train accuracy');
ax.plot(x, train_stats['Test accuracy'].values, '-b', label = 'test accuracy');

plt.legend()

def view_classify(img, ps, title):

    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize = (6, 9), ncols = 2)
    image = img.permute(1, 2, 0)
    ax1.imshow(image.numpy())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(list(classes.values()), size = 'small');
    ax2.set_title(title)
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    plt.show()

for batch_idx, (inputs, labels) in enumerate(testloader):
inputs, labels = inputs.to(device), labels.to(device)
img = inputs[0]
label_true = labels[0]
ps = model(inputs)
view_classify(img.cpu(), torch.softmax(ps[0].cpu(), dim = 0), classes[int(label_true.cpu().numpy())])

break;
