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
parser.add_argument('--finetune_batch_size', default=32, type=int, help='finetuning batch size')
parser.add_argument('--unlearn_batch', default=114, type=int, help='what batch of data should be unlearned')
parser.add_argument('--finetune_epochs', default=1, type=int, help='number of finetuning epochs')
parser.add_argument('--checkpoint_freq', default = 5, type=int, help='checkpointing frequnecy')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Getting model

#=====================================================================
#============== GETTING PRETRAINED MODEL ============= ===============
#=====================================================================
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

#=====================================================================
#==================== CONSTRUCTING ALL THE DIFFERENT DATA SETS =======
#=====================================================================
print('==> Preparing Data...')

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.finetune_batch_size, shuffle=True, num_workers=2)

transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Preparing Hessian data..')
trainset_list = list(trainset)
hessian_loader = trainset_list[:512]
hessian_loader = torch.utils.data.DataLoader(hessian_loader, batch_size=512, shuffle=False, num_workers=2)

print('==> Preparing Finetuning data...')

batch_star = trainset_list[args.finetune_batch_size * args.unlearn_batch: args.finetune_batch_size * (args.unlearn_batch+1)]
data_no_unlearned = trainset_list[:args.finetune_batch_size * args.unlearn_batch] + trainset_list[args.finetune_batch_size * (args.unlearn_batch+1):]
unlearned_loader = torch.utils.data.DataLoader(batch_star, batch_size=args.finetune_batch_size, shuffle=False, num_workers=2)

#saving the weights of the pretrained model
M_pretrain = copy.deepcopy(net)
w_pretrain_weights_tensor = [param for param in M_pretrain.parameters()]
w_pretrain_weights = weights_to_list_fast(w_pretrain_weights_tensor)

print('==> Beginning iteration over T=0 to T=500...')
data_ret = {}

for T in range(10,50):
    print('T = ', T)
    #=====================================================================
    #======================= FINETUNING M NORMALLY =======================
    #=====================================================================
    M = copy.deepcopy(M_pretrain)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(M.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    idx = 0
    data = batch_star + data_no_unlearned[:args.finetune_batch_size * (T-1)]
    data_loader = torch.utils.data.DataLoader(data, batch_size = args.finetune_batch_size, shuffle = False, num_workers = 2)
    M.train()
    rolling_sigma = []
    curr_weights = w_pretrain_weights
    curr_error = 0
    for e in range(0,args.finetune_epochs):
        for i,(inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = M(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                #get your current unlearning error:
                interim_weights_tensor = [param for param in M.parameters()]
                interim_weights = weights_to_list_fast(interim_weights_tensor)
                if i==0:
                    curr_weights = w_pretrain_weights
                    iter_num = 0
                    curr_error = 0
                    rolling_sigmas = []
                    hessian_comp = hessian(M, criterion, data=(inputs, targets), cuda=True)
                    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
                    sigma = np.sqrt(top_eigenvalues[-1])
                    rolling_sigma.append(sigma)
                if i%args.checkpoint_freq != 0:
                    hessian_comp = hessian(M, criterion, data=(inputs, targets), cuda=True)
                    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
                    sigma = np.sqrt(top_eigenvalues[-1])
                    rolling_sigma.append(sigma)
                    iter_num = (i%args.checkpoint_freq)/2
                    actual_sigma = sum(rolling_sigma)/len(rolling_sigma)
                    delta_weights = np.linalg.norm((np.array(interim_weights) - np.array(curr_weights)))
                    interim_error = curr_error+ (args.lr * args.lr) *delta_weights * actual_sigma * iter_num
                if i!=0 and i%args.checkpoint_freq == 0:
                    curr_weights = interim_weights
                    curr_error = interim_error
                    rolling_sigma = []
                    hessian_comp = hessian(M, criterion, data=(inputs, targets), cuda=True)
                    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
                    sigma = np.sqrt(top_eigenvalues[-1])
                    rolling_sigma.append(sigma)
                    interim_error = curr_error

    print('==> Done finetuning M...')
    unlearning_error = interim_error
    sigma = sigma
    delta_weights = delta_weights
    M_weights_tensor = [param for param in M.parameters()]
    w_M_weights = weights_to_list_fast(M_weights_tensor)
    
    #=====================================================================
    #======================= RETRAINING M TO GET M' ======================
    #=====================================================================
    M_retrain = copy.deepcopy(M_pretrain)
    criterion = nn.CrossEntropyLoss()
    optimizer_retrain = optim.SGD(M_retrain.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    scheduler_retrain = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retrain, T_max=200)
    data_retrain = data_no_unlearned[:args.finetune_batch_size * (T-1)]
    data_retrain_loader = torch.utils.data.DataLoader(data_retrain, batch_size = args.finetune_batch_size, shuffle = False, num_workers = 2)
    M_retrain.train()
    for e in range(0,args.finetune_epochs):
        for i,(inputs, targets) in enumerate(data_retrain_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_retrain.zero_grad()
                outputs = M_retrain(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_retrain.step()
    print('==> Done retraining M...')
    M_retrain_tensor = [param for param in M_retrain.parameters()]
    w_M_retrain_weights = weights_to_list_fast(M_retrain_tensor)
    #=====================================================================
    #======================= gradient ascent time ========================
    #=====================================================================
    M_unlearned = copy.deepcopy(M)
    optimizer_unlearned = optim.SGD(M_unlearned.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_unlearned, T_max=200)

    M_unlearned.train()

    for i,(img,label) in enumerate(unlearned_loader):
        img = img.cuda()
        label = label.cuda()
        output = M_unlearned(img)
        loss = criterion(output,label)
        loss.backward(retain_graph=True)
        grads = torch.autograd.grad(loss, [param for param in M_unlearned.parameters()],create_graph = True)
    old_params = {}
    for i, (name, params) in enumerate(M.named_parameters()):
        old_params[name] = params.clone()
        old_params[name] += args.finetune_epochs * args.lr * grads[i]
    for name, params in M_unlearned.named_parameters():
        params.data.copy_(old_params[name])
    M_unlearned_tensor = [param for param in M_unlearned.parameters()]
    w_M_unlearned_weights = weights_to_list_fast(M_unlearned_tensor)
    #=====================================================================
    #================= compute verification error ========================
    #=====================================================================
    verification_error = np.linalg.norm((np.array(w_M_retrain_weights) - np.array(w_M_unlearned_weights)))
    ret = {}
    ret['sigma'] = sigma
    ret['delta weights'] = delta_weights
    ret['T'] = (T*args.finetune_epochs)-1
    ret['verification error'] = verification_error
    ret['unlearning error'] = unlearning_error
    data_ret[(T,idx)] = ret


import pickle
if not os.path.isdir('cum_correlation_results'):
    os.mkdir('cum_correlation_results')
path = './cum_correlation_results/resnet_epochs=' +str(args.epochs) + '_pretrain_BS='+str(args.pretrain_batch_size) +'_cumulative_freq='+str(args.checkpoint_freq)+'_finetuneEpoch='+str(args.finetune_epochs) +'_finetuneBS='+str(args.finetune_batch_size)+'_unlearn_batch='+str(args.unlearn_batch) +'.p'
pickle.dump(data_ret, open(path, 'wb'))


