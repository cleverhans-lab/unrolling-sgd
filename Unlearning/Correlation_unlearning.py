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


def validate(model,loader):
  total = 0
  correct = 0
  for imgs, labels in loader:
    batch_size = len(imgs)
    total += batch_size
    imgs, labels = imgs.to(device), labels.to(device)
    out_probs = model(imgs)
    out = torch.argmax(out_probs, dim=1)
    labels = labels.view(out.shape)
    correct += torch.sum(out == labels)
  return correct, total

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

parser.add_argument('--model', default='resnet', type=str, help='resnet or vgg')
parser.add_argument('--loss_func', default='regular', type=str, help='loss function: regular,hessian, hessianv2, std_loss')
parser.add_argument('--dataset', default = 'cifar10', type=str, help ='cifar10, cifar100')
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--finetune_batch_size', default=32, type=int, help='finetuning batch size')
parser.add_argument('--unlearn_batch', default=114, type=int, help='what batch of data should be unlearned')
parser.add_argument('--finetune_epochs', default=1, type=int, help='number of finetuning epochs')
parser.add_argument('--regularizer', default=0.0, type=float, help='number of finetuning epochs')
parser.add_argument('--l2_regularizer', default=0.0, type=float, help='number of finetuning epochs')
parser.add_argument('--eval_every', default=10, type=float, help='eval every')

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

def my_cross_entropy(x, y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()

    #x is (N,C)
    N,C = x.shape
    p = F.softmax(x,1)
    hessian_loss = torch.sum(p *(1-p),dim =1)
    hessian_loss = hessian_loss.mean()

    loss = loss + std_reg * hessian_loss
    return loss

from torchvision import transforms, datasets
import torch.nn as nn

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
    net = ResNet18(num_classes)
if args.model == 'vgg':
    net = VGG('VGG19',num_classes)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.pretrain_batch_size, shuffle= True)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr = 0.01,weight_decay = args.l2_regularizer)
for ep in range(0,args.pretrain_epochs):
  print(f'on pretraining epoch = {ep}')
  for i,(imgs,labels) in enumerate(train_loader):
    imgs = imgs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    out = net(imgs)
    if args.loss_func == 'hess':
        loss = my_cross_entropy(out,labels)
    if args.loss_func =='std':
        loss = std_loss(out,labels)
    if args.loss_func == 'regular':
        loss = loss_fn(out,labels)
    loss.backward()
    optimizer.step()

test_loader = torch.utils.data.DataLoader(testset,args.pretrain_batch_size,shuffle = False)
correct, total = validate(net, test_loader)
print(f"Testing accuracy after {args.pretrain_epochs} epoch of pretraining = {100*correct/total}")
final_test_acc  = correct/total

#Saving pretrained stuff
#saving the weights of the pretrained model
M_pretrain = copy.deepcopy(net)
w_pretrain_weights_tensor = [param for param in M_pretrain.parameters()]
w_pretrain_weights = weights_to_list_fast(w_pretrain_weights_tensor)


M = copy.deepcopy(M_pretrain)
M_retrained = copy.deepcopy(M_pretrain)

M.train()
M_retrained.train()

lr = 0.01

criterion_unl = nn.CrossEntropyLoss()
#initialize the loss functions, optimizers and schedulers for both models
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(M.parameters(), lr= lr,weight_decay = args.l2_regularizer)


criterion_retrain = nn.CrossEntropyLoss()
optimizer_retrain = optim.SGD(M_retrained.parameters(), lr=lr,weight_decay = args.l2_regularizer)

trainset_list = list(trainset)
batch_star = trainset_list[:args.finetune_batch_size]
unlearned_loader = torch.utils.data.DataLoader(batch_star, batch_size = args.finetune_batch_size, shuffle=False, num_workers=2)
#the data that we are finetuning on (with x* at the beginning)
data_loader = torch.utils.data.DataLoader(trainset, batch_size = args.finetune_batch_size, shuffle= False, num_workers =2)
#we need some lists for statistics
sigma_list = []
delta_weights_list = []
unl_error_list = []
rolling_unl_error_list = []
ver_error_list = []
hessian_criterion  = nn.CrossEntropyLoss()



grad_list = []
for ep in range(0,args.finetune_epochs):
    for main_idx,(inputs, targets) in enumerate(data_loader):
        M.train()
        print('Epoch = ',ep, 'on t = ',main_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        if main_idx ==0:
            #only update M:
            optimizer.zero_grad()
            outputs_M = M(inputs)
            if args.loss_func == 'hess':
                loss_M = my_cross_entropy(outputs_M,targets)
            if args.loss_func =='std':
                loss_M = std_loss(outputs_M,targets)
            if args.loss_func =='regular':
                loss_M = loss_fn(outputs_M,targets)
            loss_M.backward()
            optimizer.step()
            #now we also want to compute the gradient:
            
            for i,(img,label) in enumerate(unlearned_loader):
                img = img.cuda()
                label = label.cuda()
                output_grad = M(img)
                if args.loss_func == 'hess':
                    loss_grad = my_cross_entropy(output_grad,label)
                if args.loss_func =='std':
                    loss_grad = std_loss(output_grad,label)
                if args.loss_func =='regular':
                    loss_grad = loss_fn(output_grad,label)
                loss_grad.backward(retain_graph=True)
                grads = torch.autograd.grad(loss_grad, [param for param in M.parameters()],create_graph = True)
                grad_list.append(grads)
        if main_idx !=0:
            #update both M and M'
            optimizer.zero_grad()
            outputs_M = M(inputs)
            if args.loss_func == 'hess':
                loss_M = my_cross_entropy(outputs_M,targets)
            if args.loss_func =='std':
                loss_M = std_loss(outputs_M,targets)
            if args.loss_func =='regular':
                loss_M = loss_fn(outputs_M,targets)
            loss_M.backward()
            optimizer.step()

            optimizer_retrain.zero_grad()
            outputs_retrain = M_retrained(inputs)
            if args.loss_func == 'hess':
                loss_retrain = my_cross_entropy(outputs_retrain,targets)
            if args.loss_func =='std':
                loss_retrain = std_loss(outputs_retrain,targets)
            if args.loss_func =='regular':
                loss_retrain = loss_fn(outputs_retrain,targets)
            loss_retrain.backward()
            optimizer_retrain.step()
        if main_idx%args.eval_every == 0:
            print('measuring stuff!!!!')
            M.eval()
            if args.loss_func == 'hess':
                hessian_comp = hessian(M, my_cross_entropy, data=(inputs, targets), cuda=True)
            if args.loss_func =='std':
                hessian_comp = hessian(M, std_loss, data=(inputs, targets), cuda=True)
            if args.loss_func =='regular':
                hessian_comp = hessian(M, loss_fn, data=(inputs, targets), cuda=True)
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
            optimizer_unlearned = optim.SGD(M_unlearned.parameters(), lr = lr,weight_decay = args.l2_regularizer)

            M_unlearned.train()
            old_params = {}
            for i, (name, params) in enumerate(M.named_parameters()):
                old_params[name] = params.clone()
                for grads in grad_list:
                    old_params[name] += lr * grads[i]
            for name, params in M_unlearned.named_parameters():
                params.data.copy_(old_params[name])
            M_unlearned_tensor = [param for param in M_unlearned.parameters()]
            w_M_unlearned_weights = weights_to_list_fast(M_unlearned_tensor)

            #Now that we have the 3 models, M_(N+t), M'_(N+t) and M''_(N+t), let's compute the unlearning error of M. We will do this two ways: one with a running average sigma, one without
            delta_weights = np.linalg.norm((np.array(w_M_weights) - np.array(w_pretrain_weights)))
            #print('wow, sigma is = ',sigma)
            unl_error = (lr * lr) *((len(data_loader)*ep)+ main_idx) *(1/2) * delta_weights *sigma
            rolling_unl_error = (lr * lr) *((len(data_loader)*ep)+ main_idx) *(1/2) * delta_weights * (sum(sigma_list)/len(sigma_list))
            #now compute the verification error
            verification_error = np.linalg.norm((np.array(w_M_retrain_weights) - np.array(w_M_unlearned_weights)))
            delta_weights_list.append(delta_weights)
            unl_error_list.append(unl_error)
            rolling_unl_error_list.append(rolling_unl_error)
            ver_error_list.append(verification_error)
        #if main_idx == 50:
        #    break
ret = {}
ret['test_acc'] = final_test_acc
ret['sigma'] = sigma_list
ret['delta weights'] = delta_weights_list
ret['verification error'] = ver_error_list
ret['unlearning error'] = unl_error_list
ret['rolling unlearning error'] = rolling_unl_error_list

import pickle
if not os.path.isdir('amnesiac_loss_results'):
    os.mkdir('amnesiac_loss_results')
path = f'./amnesiac_loss_results/{args.model}_{args.dataset}_{args.loss_func}_{args.pretrain_batch_size}_{args.pretrain_epochs}_{args.finetune_batch_size}_{args.finetune_epochs}_{args.regularizer}_{args.l2_regularizer}_{args.eval_every}.p'
pickle.dump(ret, open(path, 'wb'))
