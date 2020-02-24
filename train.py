from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import os
import argparse
import time
import wget
import zipfile

from data_prepare import *
from trojan_resnet import TrojanResNet18, TrojanResNet34, TrojanResNet50, TrojanResNet101, TrojanResNet152

parser = argparse.ArgumentParser(description='Train CIFAR model')
parser.add_argument('--data_root', type=str, default='data', help='the root address of datasets')
parser.add_argument('--save_dir', type=str, default='checkpoint', help='model output directory')
parser.add_argument('--saved_model', type=str, default='', help='load from saved model and test only')
parser.add_argument('--model', type=str, default='trojan_resnet18', help='type of model (TrojanResnet18 / TrojanResnet34 / TrojanResnet50 / TrojanResnet101 / TrojanResnet152)')
parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--datasets_name', type=str, nargs='+', default='cifar10', help='array of dataset names selected from {cifar10, cifar100, svhn, gtsrb}')
parser.add_argument('--norm_type', type=str, default='batch_norm', help='type of normalization (group_norm / batch_norm)')
parser.add_argument('--seed', type=int, default=0, help='initial seed for randomness')
args = parser.parse_args()

print(args)

num_classes = []
max_num_classes = 0

def generate_dataloader(dataset):
    global max_num_classes
    global num_classes
    
    if not os.path.exists(args.data_root):
        os.mkdir(args.data_root)

    if dataset == 'cifar10':
        if max_num_classes < 10:
            max_num_classes = 10
        num_classes.append(10)
        transform_train = cifar_transform_train
        transform_test = cifar_transform_test
        trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        if max_num_classes < 10:
            max_num_classes = 10
        num_classes.append(10)
        transform_train = cifar_transform_train
        transform_test = cifar_transform_test
        trainset = torchvision.datasets.SVHN(root=args.data_root, split='train', download=True, transform=transform_train) + torchvision.datasets.SVHN(root=args.data_root, split='extra', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=args.data_root, split='test', download=True, transform=transform_test)
    elif dataset == 'cifar100':
        if max_num_classes < 100:
            max_num_classes = 100
        num_classes.append(100)
        transform_train = cifar_transform_train
        transform_test = cifar_transform_test
        trainset = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=transform_test)
    elif dataset == 'gtsrb':
        if not os.path.exists(os.path.join(args.data_root, "GTSRB")):
            wget.download("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip", args.data_root)
            with zipfile.ZipFile(os.path.join(args.data_root, "GTSRB-Training_fixed.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(args.data_root))
            gtsrb_initialize_data(args.data_root)
        if max_num_classes < 43:
            max_num_classes = 43
        num_classes.append(43)
        transform_train = gtsrb_transform_train
        transform_test = gtsrb_transform_test
        trainset = torchvision.datasets.ImageFolder(args.data_root + '/GTSRB/Training', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(args.data_root + '/GTSRB/Test', transform=transform_test)
    else:
        print('invalid dataset ' + dataset + '!')
        exit()
    

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

#generate dataloaders
datasets_name = args.datasets_name
datasets = []
for i in range(len(datasets_name)):
    datasets += [datasets_name[i]] 

trainloaders = []
testloaders = []
for i in range(len(datasets)):
    trainloader, testloader = generate_dataloader(datasets[i])
    trainloaders.append(trainloader)
    testloaders.append(testloader)

#initialize the neural network
linear_base = IMG_SIZE * IMG_SIZE / 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.model == 'trojan_resnet18':
    net = TrojanResNet18(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
elif args.model == 'trojan_resnet34':
    net = TrojanResNet34(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
elif args.model == 'trojan_resnet50':
    net = TrojanResNet50(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
elif args.model == 'trojan_resnet101':
    net = TrojanResNet50(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)
elif args.model == 'trojan_resnet152':
    net = TrojanResNet50(seed=0, num_classes=max_num_classes, linear_base=linear_base, norm_type=args.norm_type)

if args.saved_model != '':
    checkpoint = torch.load(args.saved_model)
    net.load_state_dict(checkpoint['net'])
    
net = net.to(device)
for i in range(len(trainloaders)):
    net.reset_seed(200 * i + args.seed)

#setting for training
torch.manual_seed(int(time.time()))
criterion = nn.CrossEntropyLoss()
test_criterion = nn.NLLLoss()
best_acc = [0] * len(datasets)
lr = args.lr
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

def train(epoch, optimizer):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = [0] * len(trainloaders)
    correct = [0] * len(trainloaders)
    total = [0] * len(trainloaders)

    iter_list = []
    for i in range(len(trainloaders)):
        iter_list.append(iter(enumerate(trainloaders[i])))

    stop = False
    dataset_id = 0
    while not stop:
        optimizer.zero_grad()
        for i in range(len(trainloaders)):
            try:
                batch_idx, (inputs, targets) = next(iter_list[i])
                inputs, targets = inputs.to(device), targets.to(device)
                net.reset_seed(i * 200 + args.seed)
                outputs = net(inputs)[:, :num_classes[i]]
                loss = criterion(outputs, targets)
                loss.backward()
                train_loss[i] += loss.item()
                _, predicted = outputs.max(1)
                total[i] += targets.size(0)
                correct[i] += predicted.eq(targets).sum().item()
            except StopIteration:
                stop = True
        optimizer.step()
    for i in range(len(trainloaders)):
        print(datasets[i] + ' ==>>> train loss: {:.6f}, accuracy: {:.4f}'.format(train_loss[i]/(batch_idx + 1), 100.*correct[i]/total[i]))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = [0] * len(testloaders)
    correct = [0] * len(testloaders)
    total = [0] * len(testloaders)
    
    for i in range(len(testloaders)):
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloaders[i]):
                inputs, targets = inputs.to(device), targets.to(device)
                net.reset_seed(i * 200 + args.seed)
                outputs = F.softmax(net(inputs), dim=1)[:, :num_classes[i]]
                loss = test_criterion(outputs.log(), targets)
                test_loss[i] += loss.item()
                _, predicted = outputs.max(1)
                total[i] += targets.size(0)
                correct[i] += predicted.eq(targets).sum().item()
            print(datasets[i] + ' ==>>> test loss: {:.6f}, accuracy: {:.4f}'.format(test_loss[i]/(batch_idx+1), 100.*correct[i]/total[i]))
        # Save checkpoint.
        acc = 100.*correct[i]/total[i]
        if acc > best_acc[i]:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            dataset_name_all = datasets[0]
            for i in range(len(datasets) - 1):
                dataset_name_all += '_'
                dataset_name_all += datasets[i + 1]
            torch.save(state, '%s/mix_%s_%s_seed_%d.pth' % (args.save_dir, dataset_name_all, args.model, args.seed))
            best_acc[i] = acc

if args.saved_model != '':
    test(checkpoint['epoch'])
    exit()
        
optimizer = optim.Adam(net.parameters(), lr=args.lr)
first_drop, second_drop = False, False
for epoch in range(args.epochs):
    train(epoch, optimizer)
    test(epoch)
    if (not first_drop) and (epoch+1) >= 0.5 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        first_drop = True
    if (not second_drop) and (epoch+1) >= 0.75 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        second_drop = True

for i in range(len(best_acc)):
    print(best_acc[i])

state = {
    'net': net.state_dict(),
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
    
dataset_name_all = datasets[0]
for i in range(len(datasets) - 1):
    dataset_name_all += '_'
    dataset_name_all += datasets[i + 1]
torch.save(state, '%s/mix_%s_%s_seed_%d.pth' % (args.save_dir, dataset_name_all, args.model, args.seed))
