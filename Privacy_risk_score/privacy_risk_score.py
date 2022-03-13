import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import pickle
import os
import argparse
import copy
import numpy as np

from PyHessian.pyhessian import hessian # Hessian computation
from ResNet_CIFAR10 import *
from helper_membership_inference import *

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

parser = argparse.ArgumentParser(description='Membership inference results')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--unlearn_batch', default=32, type=int, help='what batch of data should be unlearned')
parser.add_argument('--finetune_epochs', default=1, type=int, help='number of finetuning events')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing pretraining data..')
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Preparing Hessian data..')
trainset_list = list(trainset)
hessian_loader = trainset_list[:512]
hessian_loader = torch.utils.data.DataLoader(hessian_loader, batch_size=512, shuffle=False, num_workers=2)

print('==> Preparing Finetuning data...')
        
batch_star = trainset_list[args.batch_size * args.unlearn_batch: args.batch_size * (args.unlearn_batch+1)]
data_no_unlearned = trainset_list[:args.batch_size * args.unlearn_batch] + trainset_list[args.batch_size * (args.unlearn_batch+1):]
unlearned_loader = torch.utils.data.DataLoader(batch_star, batch_size=args.batch_size, shuffle=False, num_workers=2)

training_no_unlearned = torch.utils.data.DataLoader(data_no_unlearned, batch_size=args.batch_size, shuffle=True, num_workers=2)
#We have 4 loaders:
#trainloader -> the trainloader for pretraining
#testloader -> the loader for testing pretraining (sort of optional)
#hessian_loader -> a loader with 1 batch of 512 points to be used for the singular value calculation
#unlearned_loader -> the loader that has the 1 batch we want to unlearn

#2 very important lists:
# data_no_unlearned: The original training set but without the batch we want to unlearn
# batch_star: The data corresponding to the batch we want to unlearn but in a list format



print('==> Building model..')
net = ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    for batch_idx, (inputs, targets) in enumerate(training_no_unlearned):
        if batch_idx != args.unlearn_batch:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
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
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Final testing Accuracy: %.3f%% (%d/%d)' %(100.*correct/total, correct, total))
    acc = 100.*correct/total
    return acc

print('==> Beginning Pretraining....')
for epoch in range(0, args.epochs):
    train(epoch)
    scheduler.step()
acc =test(0)

print('==> Done pretraining now saving model...')
state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': args.epochs,
            'batch size': args.batch_size,
        }
if not os.path.isdir('pretrained_models'):
    os.mkdir('pretrained_models')
torch.save(state, './pretrained_models/resnet_Pretrain_epochs='+ str(args.epochs).zfill(3) +'_batch_size=' + str(args.batch_size).zfill(3) + '_Unlearned_batch=' + str(args.unlearn_batch).zfill(4) + '.pth')
M_pretrain = copy.deepcopy(net)
w_pretrain_weights_tensor = [param for param in M_pretrain.parameters()]
w_pretrain_weights = weights_to_list_fast(w_pretrain_weights_tensor)


train_probs = np.load('./train_probs.npy')
val_probs = np.load('./val_probs.npy')

train_divisions = np.load('./train_divisions.npy')
val_divisions = np.load('./val_divisions.npy')

total_ret = {}
print('==> Beginning big sweep...')
for T in range(5,500,1):
        idx = 0
        M = copy.deepcopy(M_pretrain)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(M.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        #1. Build finetuning dataset
        if idx == 0:
            data = batch_star + data_no_unlearned[:args.batch_size * (T-1)]
        if idx == T-1:
            data = data_no_unlearned[:args.batch_size *(T-1)] + batch_star
        else:
            data = data_no_unlearned[:args.batch_size * idx] + batch_star + data_no_unlearned[(idx) *args.batch_size:args.batch_size *(T-1)] 
        data_loader = torch.utils.data.DataLoader(data, batch_size = args.batch_size, shuffle = False, num_workers = 2)
        M.train()
        for e in range(0, args.finetune_epochs):
            for i,(inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = M(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        print('==> Done finetuning M...')
        M_weights_tensor = [param for param in M.parameters()]
        w_M_weights = weights_to_list_fast(M_weights_tensor)
        print('==> Getting Unlearning error...')    
        for i,(img,label) in enumerate(hessian_loader):
            img = img.cuda()
            label = label.cuda()
            
            delta_weights = np.linalg.norm((np.array(w_M_weights) - np.array(w_pretrain_weights)))
            hessian_comp = hessian(M, criterion, data=(img, label), cuda=True)
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
            sigma = np.sqrt(top_eigenvalues[-1])
        unlearning_error = (args.lr * args.lr) *((T*args.finetune_epochs)-1) *(1/2) * delta_weights *sigma #* (args.finetune_batch_size/60000)
        
        print('==> Unlearning M to get M_unlearned...')
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
            old_params[name] += args.finetune_epochs* args.lr * grads[i]
        for name, params in M_unlearned.named_parameters():
            params.data.copy_(old_params[name])

        print('==> Getting Privacy Risk scores...')

        privacy_risk_M = Get_risks(unlearned_loader,args.batch_size,M)
        privacy_risk_unlearned = Get_risks(unlearned_loader, args.batch_size, M_unlearned)

        M_counts, second = Get_Counts(privacy_risk_M, privacy_risk_M)
        unlearned_counts, second_second = Get_Counts(privacy_risk_unlearned, privacy_risk_unlearned)
        ret = {}
        ret['M_counts'] = M_counts
        ret['unlearning error'] = unlearning_error
        ret['unlearned_counts'] = unlearned_counts
        ret['M privacy risk scores'] = privacy_risk_M
        ret['unlearned privacy risk scores'] = privacy_risk_unlearned
        total_ret[(T,idx)] = ret

total_ret['acc'] = acc
path = os.path.abspath(os.getcwd())
filename = path+ '/new_MI_results/Pretrain_epochs='+str(args.epochs).zfill(3) +'_batch_size=' + str(args.batch_size).zfill(3) +'_finetune_epochs='+str(args.finetune_epochs).zfill(2)+ '_Unlearned_batch=' + str(args.unlearn_batch).zfill(4) + '.p'
pickle.dump(total_ret, open(filename, 'wb'))
