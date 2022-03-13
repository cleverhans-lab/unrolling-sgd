import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PyHessian.pyhessian import hessian # Hessian computation
from ResNet_CIFAR10 import *
from VGG_model import *
import os
import argparse
torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)

def validate(model,loader):
  total = 0
  correct = 0
  model.eval()
  for imgs, labels in loader:
    batch_size = len(imgs)
    total += batch_size
    imgs, labels = imgs.to(device), labels.to(device)
    out_probs = model(imgs)
    out = torch.argmax(out_probs, dim=1)
    labels = labels.view(out.shape)
    correct += torch.sum(out == labels)
  return correct, total




parser = argparse.ArgumentParser(description='Finetuning for verification error')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('--model', default='resnet', type=str, help='resnet or vgg')
parser.add_argument('--dataset', default = 'cifar10', type=str, help ='cifar10, cifar100')
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--regularizer', default=0.0, type=float, help='number of finetuning epochs')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

std_reg =args.regularizer

def std_loss(x,y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    avg_std = torch.sum(torch.std(x, dim=1))/(len(x.view(-1)))
    loss = loss + std_reg*avg_std
    return loss



if args.dataset == 'cifar10':
    num_classes = 10
    transform_train = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
if args.dataset == 'cifar100':
    num_classes = 100
    transform_train = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),])
    transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

if args.model == 'resnet':
    target_net = ResNet18(num_classes)
#if args.model == 'resnet50':
#    target_net = ResNet50(num_classes)
#if args.model == 'resnet101':
#    target_net = ResNet101(num_classes)
if args.model == 'vgg':
    target_net = VGG('VGG19',num_classes)
target_net = target_net.to(device)

if device == 'cuda':
    target_net = torch.nn.DataParallel(target_net)
    cudnn.benchmark = True

#First train target model
test_loader = torch.utils.data.DataLoader(testset,args.pretrain_batch_size,shuffle = False)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.pretrain_batch_size, shuffle= True)
print(f'length of target model dataset is {len(train_loader)}')
optimizer = optim.SGD(target_net.parameters(), lr = 0.01,momentum=0.9, weight_decay=5e-4)

for ep in range(0,args.pretrain_epochs):
  print(f'Target Model Epoch = {ep}')
  for i,(imgs,labels) in enumerate(train_loader):
    imgs = imgs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    out = target_net(imgs)
    loss = std_loss(out,labels)
    loss.backward()
    optimizer.step()

correct, total = validate(target_net, test_loader)
print(f"Testing accuracy after {args.pretrain_epochs} epoch of training target model = {100*correct/total}")

#saving target model:
state = {
            'net': target_net.state_dict(),
            'acc': 100*(correct/total),
        }
if not os.path.isdir('MI_models'):
    os.mkdir('MI_models')
torch.save(state, f'./MI_models/Target_model={args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}.pth')


#On to shadow model

trainset_list = list(trainset)
half_loader = torch.utils.data.DataLoader(trainset_list[:int(0.5*len(trainset_list))], batch_size = args.pretrain_batch_size, shuffle= True)
print(f'length of shadow model dataset is {len(half_loader)}')
if args.model == 'resnet':
    shadow_net = ResNet18(num_classes)
#if args.model == 'resnet50':
#    shadow_net = ResNet50(num_classes)
#if args.model == 'resnet101':
#    shadow_net = ResNet101(num_classes)
if args.model == 'vgg':
    shadow_net = VGG('VGG19',num_classes)
shadow_net = shadow_net.to(device)

if device == 'cuda':
    shadow_net = torch.nn.DataParallel(shadow_net)
    cudnn.benchmark = True

shadow_optimizer = optim.SGD(shadow_net.parameters(), lr = 0.01,momentum=0.9, weight_decay=5e-4)
for ep in range(0,args.pretrain_epochs):
  print(f'Shadow Model Epoch = {ep}')
  for i,(imgs,labels) in enumerate(half_loader):
    imgs = imgs.to(device)
    labels = labels.to(device)
    shadow_optimizer.zero_grad()
    out = shadow_net(imgs)
    loss = std_loss(out,labels)
    loss.backward()
    shadow_optimizer.step()

correct, total = validate(shadow_net, test_loader)
print(f"Testing accuracy after {args.pretrain_epochs} epoch of training shadow model = {100*correct/total}")

#saving target model:
state = {
            'net': shadow_net.state_dict(),
            'acc': 100*(correct/total),
        }
if not os.path.isdir('MI_models'):
    os.mkdir('MI_models')
torch.save(state, f'./MI_models/Shadow_model={args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}.pth')
