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

import os
import argparse

#=====================================================================
#============== ANVITHS HELPERS FOR L2 NORM OF WEIGHTS ===============
#=====================================================================
def set_weights_fast(x, weights):
  with torch.no_grad():
    start = 0
    #index = 0
    for weight in weights:
      length = len(weight.view(-1))
      array = x[start:start+length]
      weight_new = torch.Tensor(array).view(*weight.shape)

      weight.data.copy_(weight_new)
      #index +=1
      start += length


#puts the weights into a list, but faster
def weights_to_list_fast(weights):
  with torch.no_grad():
    weights_list = []
    for weight in weights:
      list_t = weight.view(-1).tolist()
      weights_list = weights_list + list_t

    return weights_list
#=====================================================================
#=====================================================================
#=====================================================================



parser = argparse.ArgumentParser(description='Finetuning for verification error')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--T', default=10, type=int, help='number of SGD updates')
parser.add_argument('--finetune_batch_size', default=32, type=int, help='finetuning batch size')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Getting model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('==> Loading in pre-trained model..')
assert os.path.isdir('pretrained_models'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./pretrained_models/resnet' +'_epochs=' +str(args.epochs) + '_bs='+str(args.pretrain_batch_size) +'.pth')
net.load_state_dict(checkpoint['net'])

print('==> Preparing Data...')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainset_list = list(trainset)
#getting w_pretrained
w_pretrain_tensor = [param for param in net.parameters()]
w_pretrain = weights_to_list_fast(w_pretrain_tensor)
hessian_loader = trainset_list[:250]
hessian_loader = torch.utils.data.DataLoader(
            hessian_loader, batch_size=250, shuffle=False, num_workers=2)

data = {}
data['pretrain_acc'] = checkpoint['acc']
for T in range(0,args.T):
    criterion = nn.CrossEntropyLoss()
    M = copy.deepcopy(net)
    optimizer_M = optim.SGD(M.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    scheduler_M = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_M, T_max=200)
    num_points_req =  T * args.finetune_batch_size
    finetune_trainset = trainset_list[:num_points_req]
    trainloader = torch.utils.data.DataLoader(
            finetune_trainset, batch_size=args.finetune_batch_size, shuffle=False, num_workers=2)
    print('The following two numbers should be equal: ', T, len(trainloader))

    
    #print('==> Getting Model M...')
    M.train()
    for i,(img,label) in enumerate(trainloader):
        img = img.cuda()
        label = label.cuda()
        output = M(img)
        loss = criterion(output,label)
        optimizer_M.zero_grad()
        loss.backward()
        optimizer_M.step()
    for i,(img,label) in enumerate(hessian_loader):
        img = img.cuda()
        label = label.cuda()
        M_weights = [param for param in M.parameters()]
        M_weights = weights_to_list_fast(M_weights)
        delta_weights = np.linalg.norm((np.array(M_weights) - np.array(w_pretrain)))
        hessian_comp = hessian(M, criterion, data=(img, label), cuda=True)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
        sigma = np.sqrt(top_eigenvalues[-1])
    unlearning_error = (args.lr * args.lr) *(args.T-1) *(1/2) * delta_weights *sigma #* (args.finetune_batch_size/60000)


    for k in range(0,T):
        #print('Number of SGD updates: ',T, 'Batch to unlearn: ', k)

        M_retrain = copy.deepcopy(net)
        criterion = nn.CrossEntropyLoss()
        optimizer_retrain = optim.SGD(M_retrain.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retrain, T_max=200)

        #print('==> Getting Model M_retrained...')
        M_retrain.train()
        for i,(img,label) in enumerate(trainloader):
            if i != k:
                img = img.cuda()
                label = label.cuda()
                output = M_retrain(img)
                loss = criterion(output,label)
                optimizer_retrain.zero_grad()
                loss.backward()
                optimizer_retrain.step()
        M_retrain_weights = [param for param in M_retrain.parameters()]
        M_retrain_weights = weights_to_list_fast(M_retrain_weights)
        
        M_unlearned = copy.deepcopy(net)
        optimizer_unlearned = optim.SGD(M_unlearned.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_unlearned, T_max=200)
        #print('==> Getting unlearned model M...')
        #let's calculate gradient
        M_unlearned.train()
        for i,(img,label) in enumerate(trainloader):
            if i == k:
                img = img.cuda()
                label = label.cuda()
                output = M_unlearned(img)
                loss = criterion(output,label)
                loss.backward(retain_graph=True)
                grads = torch.autograd.grad(loss, [param for param in M_unlearned.parameters()],create_graph = True)
        old_params = {}
        for i, (name, params) in enumerate(M.named_parameters()):
                old_params[name] = params.clone()
                old_params[name] += args.lr * grads[i]
        for name, params in M_unlearned.named_parameters():
                params.data.copy_(old_params[name])

        #print('==> Determining verification error...')
        w_unlearned = [param for param in M_unlearned.parameters()]
        w_unlearned = weights_to_list_fast(w_unlearned)

        verification_error = np.linalg.norm((np.array(M_retrain_weights) - np.array(w_unlearned)))
        print('Number of SGD updates: ',T, 'Batch to unlearn: ', k, 'Unlearning Error: ', unlearning_error, 'Verification error: ', verification_error)
        data[(T,k)] = (verification_error, unlearning_error)

import pickle
if not os.path.isdir('verification_results'):
    os.mkdir('verification_results')
path = './verification_results/resnet' + '_epochs=' +str(args.epochs) + '_bs='+str(args.pretrain_batch_size) +'_finetuneBS='+str(args.finetune_batch_size) +'.p'
pickle.dump(data, open(path, 'wb'))


