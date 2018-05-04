import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from capsule import *

from keras.layers import LeakyReLU, Dense

 # ####### DATA PREP ####### #
# We transform them to tensors
transform = transforms.ToTensor()
root = './data'
download = True  # download MNIST dataset or not

# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
# test_set = dset.MNIST(root=root, train=False, transform=trans)
# batch_size = 32

# train_loader = torch.utils.data.DataLoader(
#                  dataset=train_set,
#                  batch_size=batch_size,
#                  shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#                 dataset=test_set,
#                 batch_size=batch_size,
#                 shuffle=False)
kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True, **kwargs)

# ### Helpers ### #
class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='float64')[y]


# ### ------- ### #
def squash(x, axis=-1):
    s_squared_norm = x.pow(2).sum(dim=-1) + 1e-8 #epsilon
    scale = s_squared_norm.pow(2)/ (0.5 + s_squared_norm)
    # print(scale.view(-1,scale.size()[1],1).size())
    # print(x.size())
    scale = scale.view(-1,10,1)
    return scale * x

class SeqCap(nn.Module):
    def __init__(self):
        super(SeqCap, self).__init__()
        self.num_capsule = 10
        self.dim_capsule = 16
        self.dropout_prob = 0.25

        # self.fc0 = nn.Linear(784,32)
        self.fc1 = nn.Linear(32,160)
        # self.fc2 = nn.Linear(32,160)
        self.fc3 = nn.Linear(160,10)


    def forward(self,x):
        x =  Capsule(self.num_capsule, self.dim_capsule)(x)
        return F.softmax(torch.sum(x,2))
        # return F.softmax(x)

    def name(self):
        return "SeqCap"

def main():
    model = SeqCap()
    # if torch.cuda.is_available():
    #     model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    ceriation = nn.CrossEntropyLoss()
    for epoch in range(1,13):
        # trainning
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()

            x, target = Variable(x), Variable(target)
            x = x.view(-1,1,784)
            out = model(x)
            loss = ceriation(out, target)
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx+1, ave_loss))

    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):

        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        x = x.view(-1,1,784)
        out = model(x)
        loss = ceriation(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))


if __name__ == '__main__':
    main()

    # torch.save(model.state_dict(), model.name())
