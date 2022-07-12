import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
import time
import argparse
import resnet
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='sgd batch size')
parser.add_argument('--decay', default=0., type=float, help='weight decay')
parser.add_argument('--shrink', default=0.4, type=float, help='shrinkage parameter')
parser.add_argument('--perturb', default=0.1, type=float, help='noise scale')
parser.add_argument('--model', default='resnet', type=str, help='model architecture')
parser.add_argument('--data', default='cifar10', type=str, help='dataset')
parser.add_argument('--opt', default='adam', type=str, help='optimizer')
parser.add_argument('--n_samples', default=1000, type=float, help='number of samples to add to training set at each round')
parser.add_argument('--lr_2', default=-1, type=float, help='learning rate for second round (only used if n_samples < 1)')
parser.add_argument('--batch_size_2', default=-1, type=int, help='sgd batch size for second round (only used if n_samples < 1)')
parser.add_argument('--decay_2', default=-1., type=float, help='weight decay for second round (only used if n_samples < 1)')
args = parser.parse_args()

# in two-phase experiments, if left unspecified, second-round optimization parameters equal first-round parameters
if args.lr_2 < 0: args.lr_2 = args.lr
if args.batch_size_2 < 0: args.batch_size_2 = args.batch_size
if args.decay_2 < 0: args.decay_2 = args.decay
print(str(args.__dict__), flush=True)

# make dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
num_classes = 10
if args.data == 'cifar10':
    trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
if args.data == 'cifar100':
    trainset = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)
    num_classes = 100
if args.data == 'svhn':
    trainset = datasets.SVHN(root='data', split='train', download=True, transform=transform)
    testset = datasets.SVHN(root='data', split='test', download=True, transform=transform)

# our experiments use a randomly chosen 2/3 of provided 'training data' for model fitting and the remaining 1/3 to test
subset = np.random.permutation([i for i in range(len(trainset))])[:len(trainset)]
sub_train = subset[:int(len(trainset) * 2/3.)]
sub_val   = subset[int(len(trainset) * 2/3.):]
test_loader = torch.utils.data.DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(sub_val), shuffle=False)

if args.n_samples > 1: n_samples = int(args.n_samples)
else: n_samples = int(len(sub_train) * args.n_samples)

class mlp(nn.Module):
    def __init__(self, nc=3, sz=32, num_classes=10):
        super(mlp, self).__init__()
        self.nc = nc
        self.sz = sz
        self.lm1 = nn.Linear(nc * sz * sz, 100)
        self.lm2 = nn.Linear(100, 100)
        self.lm3 = nn.Linear(100, num_classes)
    def forward(self, x):
        x = x.view(-1, self.nc * self.sz ** 2)
        return self.lm3(F.relu(self.lm2(F.relu(self.lm1(x)))))

class BuildDataset(Dataset):
    def __init__(self, trainset):
        self.data = trainset 
        
    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y, index

    def __len__(self):
        return len(self.data)

def train(epoch, net, loader):
    net.train()
    total_loss = train_loss = 0.
    criterion = nn.CrossEntropyLoss()
    totalAcc = totalSamps = count = total = correct = 0
    for batch_idx, (inputs, targets, idx) in enumerate(loader):
        count += 1
        optimizer.zero_grad()
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        total_loss += loss.data.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().data.item()
        totalAcc += correct
        totalSamps += total
        if batch_idx % 10 == 0 and batch_idx != 0:
            correct = total = train_loss = 0.
    return (totalAcc / totalSamps), (total_loss / count)

def test(epoch, net):
    test_loss = correct = total = 0
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets = targets.cuda()
            outputs = net(inputs.cuda())
            loss = criterion(outputs, targets)
            test_loss += (loss.data.item() * targets.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().data.item()
        print('Test:\t' + str(epoch) + '\t' + str(test_loss/total) +'\t' + str(100.*correct/total), flush=True)

def shrink_perturb(net, shrink, perturb):
    # using a randomly-initialized model as a noise source respects how different kinds 
    # of parameters are often initialized differently
    if args.model == 'resnet': new_init = resnet.ResNet18(num_classes=num_classes).cuda()
    if args.model == 'mlp': new_init = mlp(num_classes=num_classes).cuda()

    params1 = new_init.parameters()
    params2 = net.parameters()
    for p1, p2 in zip(*[params1, params2]):
        p1.data = deepcopy(shrink * p2.data + perturb * p1.data)
    return new_init

n_train = n_samples
current_subset = np.asarray([])
batch_size = args.batch_size
lr = args.lr
decay = args.decay
if args.model == 'resnet': net = resnet.ResNet18(num_classes=num_classes).cuda()
if args.model == 'mlp': net = mlp(num_classes=num_classes).cuda()

while len(current_subset) != len(sub_train):

    # get new samples
    selected_inds = sub_train[n_train-n_samples:n_train]
    current_subset = np.concatenate((selected_inds, current_subset)).astype(int)
    dataset = BuildDataset(trainset)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(current_subset), shuffle=False)

    # initialize network
    net = shrink_perturb(net, args.shrink, args.perturb)
    net = net.train()

    # train until reaching 99% training accuracy
    epoch = 0
    if args.opt == 'sgd': optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=decay)
    if args.opt == 'adam': optimizer = optim.Adam(net.parameters(), lr=lr,  weight_decay=decay)
    while True:
        start = time.time()
        trainAcc, trainLoss = train(epoch, net, train_loader)
        end = time.time()
        print('Train ' + str(len(current_subset)) +  
                '\t' + str(epoch) + 
                '\t' + str(trainLoss) + 
                '\t' + str(trainAcc * 100) + 
                '\t' + str(end-start), flush=True)
        if trainAcc >= 0.99: break
        epoch += 1
    net = net.eval()

    # get test performance
    test(len(current_subset), net)

    # update dataset size
    if args.n_samples <= 1:
        n_samples = len(sub_train) - n_train
        n_train = len(sub_train)
        batch_size = args.batch_size_2
        lr = args.lr_2
        decay = args.decay_2
    else:    
        n_train += n_samples
