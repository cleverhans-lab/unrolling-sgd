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

std_reg =1.0
parser = argparse.ArgumentParser(description='Finetuning for verification error')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('--model', default='resnet', type=str, help='resnet or vgg')
parser.add_argument('--loss_func', default='regular', type=str, help='loss function: regular,hessian, hessianv2, std_loss')
parser.add_argument('--dataset', default = 'cifar10', type=str, help ='cifar10, cifar100')
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--finetune_batch_size', default=32, type=int, help='finetuning batch size')
parser.add_argument('--unlearn_batch', default=114, type=int, help='what batch of data should be unlearned')
parser.add_argument('--finetune_epochs', default=1, type=int, help='number of finetuning epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#=====================================================================
#==================== CONSTRUCTING ALL THE DIFFERENT DATA SETS =======
#=====================================================================
print('==> Preparing Data...')

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.finetune_batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.finetune_batch_size, shuffle=False, num_workers=2)
    num_classes = 10

if args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize( (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, num_workers=2, batch_size=args.finetune_batch_size)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader= torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2, batch_size=args.finetune_batch_size)
    num_classes = 100

#transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.finetune_batch_size, shuffle=True, num_workers=2)

#transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Preparing Hessian data..')
trainset_list = list(trainset)
hessian_loader = trainset_list[:512]
hessian_loader = torch.utils.data.DataLoader(hessian_loader, batch_size=512, shuffle=False, num_workers=2)

print('==> Preparing Finetuning data...')

batch_star = trainset_list[args.finetune_batch_size * args.unlearn_batch: args.finetune_batch_size * (args.unlearn_batch+1)]
data_no_unlearned = trainset_list[:args.finetune_batch_size * args.unlearn_batch] + trainset_list[args.finetune_batch_size * (args.unlearn_batch+1):]
unlearned_loader = torch.utils.data.DataLoader(batch_star, batch_size=args.finetune_batch_size, shuffle=False, num_workers=2)


#Getting model

#=====================================================================
#============== GETTING PRETRAINED MODEL ============= ===============
#=====================================================================
print('==> Building model..')

if args.model == 'resnet':
    net = ResNet18(num_classes)
if args.model == 'vgg':
    net = VGG('VGG19',num_classes)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('==> Loading in pre-trained model..')
assert os.path.isdir('Final_pretrained_models'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(f'./Final_pretrained_models/{args.model}_{args.dataset}_{args.loss_func}_{args.pretrain_batch_size}_{args.pretrain_epochs}.pth')
net.load_state_dict(checkpoint['net'])



#saving the weights of the pretrained model
M_pretrain = copy.deepcopy(net)
w_pretrain_weights_tensor = [param for param in M_pretrain.parameters()]
w_pretrain_weights = weights_to_list_fast(w_pretrain_weights_tensor)

print('==> Beginning iteration over T=0 to T=500...')
data_ret = {}

#1. Initialize your 2 models: M and M'
M = copy.deepcopy(M_pretrain)
M_retrained = copy.deepcopy(M_pretrain)

M.train()
M_retrained.train()


criterion_unl = nn.CrossEntropyLoss()
#initialize the loss functions, optimizers and schedulers for both models
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(M.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

criterion_retrain = nn.CrossEntropyLoss()
optimizer_retrain = optim.SGD(M_retrained.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
scheduler_retrain = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_retrain, T_max=200)

#the data that we are finetuning on (with x* at the beginning)
data = batch_star + data_no_unlearned
data_loader = torch.utils.data.DataLoader(data, batch_size = args.finetune_batch_size, shuffle = False, num_workers = 2)

sigma_list  = []
print(' T = ',len(data_loader))

#we need some lists for statistics
sigma_list = []
delta_weights_list = []
unl_error_list = []
rolling_unl_error_list = []
ver_error_list = []
hessian_criterion  = nn.CrossEntropyLoss()
for ep in range(0,args.finetune_epochs):
    for main_idx,(inputs, targets) in enumerate(data_loader):
        M.train()
        print('Epoch = ',ep, 'on t = ',main_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        if main_idx ==0:
            #only update M:
            optimizer.zero_grad()
            outputs_M = M(inputs)
            if args.loss_func == 'regular':
                loss_M = criterion(outputs_M, targets)
            if args.loss_func == 'std':
                loss_M = std_loss(outputs_M,targets)
            loss_M.backward()
            optimizer.step()
        if main_idx !=0:
            #update both M and M'
            optimizer.zero_grad()
            outputs_M = M(inputs)
            if args.loss_func == 'regular':
                loss_M = criterion(outputs_M, targets)
            if args.loss_func == 'std':
                loss_M = std_loss(outputs_M,targets)
            loss_M.backward()
            optimizer.step()    

            optimizer_retrain.zero_grad()
            outputs_retrain = M_retrained(inputs)
            if args.loss_func == 'regular':
                loss_retrain = criterion(outputs_retrain, targets)
            if args.loss_func == 'std':
                loss_retrain = std_loss(outputs_retrain,targets)
            loss_retrain.backward()
            optimizer_retrain.step()
        M.eval()
        for i,(img,label) in enumerate(hessian_loader):
            img = img.cuda()
            label = label.cuda()
            break
        if args.loss_func == 'regular':
            hessian_comp = hessian(M, hessian_criterion, data=(img, label), cuda=True)
        if args.loss_func == 'std':
            hessian_comp = hessian(M, std_loss, data=(img, label), cuda=True)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
        sigma = np.sqrt(top_eigenvalues[-1])
        sigma_list.append(sigma)

        #Now, save the weights of both M_(N+t) and M'_(N+t)
        M_weights_tensor = [param for param in M.parameters()]
        w_M_weights = weights_to_list_fast(M_weights_tensor)

        M_retrain_tensor = [param for param in M_retrained.parameters()]
        w_M_retrain_weights = weights_to_list_fast(M_retrain_tensor)

        #Now, get M''_(N+t)

        M_unlearned = copy.deepcopy(M)
        optimizer_unlearned = optim.SGD(M_unlearned.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
        scheduler_unlearned = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_unlearned, T_max=200)

        M_unlearned.train()
        M_pretrain.train()
        for i,(img,label) in enumerate(unlearned_loader):
            img = img.cuda()
            label = label.cuda()
            output_pre = M_pretrain(img)
            if args.loss_func == 'regular':
                loss_unl = criterion(output_pre, label)
            if args.loss_func == 'std':
                loss_unl = std_loss(output_pre,label)
            loss_unl.backward(retain_graph=True)
            grads = torch.autograd.grad(loss_unl, [param for param in M_pretrain.parameters()],create_graph = True)
        old_params = {}
        for i, (name, params) in enumerate(M.named_parameters()):
            old_params[name] = params.clone()
            old_params[name] += (ep+1) * args.lr * grads[i]
        for name, params in M_unlearned.named_parameters():
            params.data.copy_(old_params[name])
        M_unlearned_tensor = [param for param in M_unlearned.parameters()]
        w_M_unlearned_weights = weights_to_list_fast(M_unlearned_tensor)

        #Now that we have the 3 models, M_(N+t), M'_(N+t) and M''_(N+t), let's compute the unlearning error of M. We will do this two ways: one with a running average sigma, one without
        #print('calculating unl error....')
        delta_weights = np.linalg.norm((np.array(w_M_weights) - np.array(w_pretrain_weights)))
        #print('wow, sigma is = ',sigma)
        unl_error = (args.lr * args.lr) *((len(data_loader)*ep)+ main_idx) *(1/2) * delta_weights *sigma
        rolling_unl_error = (args.lr * args.lr) *((len(data_loader)*ep)+ main_idx) *(1/2) * delta_weights * (sum(sigma_list)/len(sigma_list))

        #now compute the verification error
        verification_error = np.linalg.norm((np.array(w_M_retrain_weights) - np.array(w_M_unlearned_weights)))
        delta_weights_list.append(delta_weights)
        unl_error_list.append(unl_error)
        rolling_unl_error_list.append(rolling_unl_error)
        ver_error_list.append(verification_error)
        #print('rolling unlearning error is = ', rolling_unl_error)
        #print('unl error is ', unl_error)
        #print('sigma is ', sum(sigma_list)/len(sigma_list))
        #print('delta weights is', delta_weights)
ret = {}
ret['sigma'] = sigma_list
ret['delta weights'] = delta_weights_list
ret['verification error'] = ver_error_list
ret['unlearning error'] = unl_error_list
ret['rolling unlearning error'] = rolling_unl_error_list


import pickle
if not os.path.isdir('final_correlation_results'):
    os.mkdir('final_correlation_results')
path = f'./final_correlation_results/{args.model}_{args.dataset}_{args.loss_func}_{args.pretrain_batch_size}_{args.pretrain_epochs}_{args.finetune_batch_size}_{args.finetune_epochs}.p'
pickle.dump(ret, open(path, 'wb'))


