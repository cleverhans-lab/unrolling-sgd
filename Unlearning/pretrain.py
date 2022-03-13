import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from ResNet_CIFAR10 import *
from VGG_model import *


#####################################################################
################### Some Loss functions things ######################
#####################################################################
def std_loss(x,y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    avg_std = torch.sum(torch.std(x, dim=1))/(len(x.view(-1)))
    loss = loss + std_reg*avg_std
    return loss



parser = argparse.ArgumentParser(description='Pre-training CIFAR10 Models')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='resnet', type=str, help='resnet or vgg')
parser.add_argument('--loss_func', default='regular', type=str, help='loss function: regular,hessian, hessianv2, std_loss')
parser.add_argument('--dataset', default = 'cifar10', type=str, help ='cifar10, cifar100')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--std_reg', default = 0.1, type = float, help= 'regularization for std loss')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
std_reg = args.std_reg
print(f'Model = {args.model} dataset = {args.dataset} loss = {args.loss_func} std lambda = {args.std_reg}')
print('==> Preparing data..')
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 10

if args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize( (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(cifar100_training, shuffle=True, num_workers=2, batch_size=args.batch_size)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader= torch.utils.data.DataLoader(cifar100_test, shuffle=True, num_workers=2, batch_size=args.batch_size)
    num_classes = 100
    
print('==> Building model..')
if args.model == 'resnet':
    net = ResNet18(num_classes)
if args.model == 'vgg':
    net = VGG('VGG19',num_classes)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if args.loss_func == 'std':
            loss = std_loss(outputs,targets)
        if args.loss_func =='regular':
            loss = criterion(outputs,targets)        
        loss.backward()
        optimizer.step()

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if args.loss_func == 'std':
                loss = std_loss(outputs,targets)
            if args.loss_func =='regular':
                loss = criterion(outputs,targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('testing Accuracy: %.3f%% (%d/%d)' %(100.*correct/total, correct, total))
    acc = 100.*correct/total
    return acc


for epoch in range(0, args.epochs):
    train(epoch)
    scheduler.step()
    test(epoch)
acc =test(0)
print('Saving..')
state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': args.epochs,
            'batch size': args.batch_size,
        }
if not os.path.isdir('Final_pretrained_models'):
    os.mkdir('Final_pretrained_models')
torch.save(state, f'./Final_pretrained_models/{args.model}_{args.dataset}_{args.loss_func}_{args.batch_size}_{args.epochs}.pth')

