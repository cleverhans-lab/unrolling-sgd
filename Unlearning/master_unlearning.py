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
parser.add_argument('--dataset', default = 'cifar10', type=str, help ='cifar10, cifar100')
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--regularizer', default=0.0, type=float, help='number of finetuning epochs')
parser.add_argument('--eval_every', default=10, type=float, help='eval every')
parser.add_argument('--decay', default=1.0, type=float, help='decay')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = args.lr
std_reg =args.regularizer

def std_loss(x,y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    avg_std = torch.sum(torch.std(x, dim=1))/(len(x.view(-1)))
    loss = loss + std_reg*avg_std
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

optimizer = optim.SGD(net.parameters(), lr =lr)
sigma_list = []
delta_weights_list = []
rolling_unl_error_list = []
unl_error_list = []

M_pretrain = copy.deepcopy(net)
w_pretrain_weights_tensor = [param for param in M_pretrain.parameters()]
w_0_weights = weights_to_list_fast(w_pretrain_weights_tensor)
for ep in range(0,args.pretrain_epochs):
  print(f'on pretraining epoch = {ep}')
  for i,(imgs,labels) in enumerate(train_loader):
    imgs = imgs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    out = net(imgs)
    loss = std_loss(out,labels)
    loss.backward()
    optimizer.step()
    #do you want to measure stuff?
    if i%args.eval_every == 0:
            net.eval()
            hessian_comp = hessian(net, std_loss, data=(imgs, labels), cuda=True)
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
            sigma = np.sqrt(top_eigenvalues[-1])
            sigma_list.append(sigma)
            #Now, save the weights 
            M_weights_tensor = [param for param in net.parameters()]
            curr_weights = weights_to_list_fast(M_weights_tensor)
            
            delta_weights = np.linalg.norm((np.array(curr_weights) - np.array(w_0_weights)))
            unl_error = (lr * lr) *((len(train_loader)*ep)+ i) *(1/2) * delta_weights *sigma
            rolling_unl_error = (lr * lr) *((len(train_loader)*ep)+ i) *(1/2) * delta_weights * (sum(sigma_list)/len(sigma_list))
            
            #now compute the verification error
            delta_weights_list.append(delta_weights)
            unl_error_list.append(unl_error)
            rolling_unl_error_list.append(rolling_unl_error)
    if i==0 and ep%10==0 and ep !=0:
        std_reg = std_reg/ args.decay
test_loader = torch.utils.data.DataLoader(testset,args.pretrain_batch_size,shuffle = False)
correct, total = validate(net, test_loader)
print(f"Testing accuracy after {args.pretrain_epochs} epoch of pretraining = {100*correct/total}")
final_test_acc  = correct/total

ret = {}
ret['test_acc'] = final_test_acc
ret['sigma'] = sigma_list
ret['delta weights'] = delta_weights_list
ret['unlearning error'] = unl_error_list
ret['rolling unlearning error'] = rolling_unl_error_list

import pickle
if not os.path.isdir('final_ever_loss_results'):
    os.mkdir('final_ever_loss_results')
path = f'./final_ever_loss_results/{args.model}_{args.dataset}_{args.decay}_{args.lr}_{args.pretrain_batch_size}_{args.pretrain_epochs}_{args.regularizer}_{args.eval_every}.p'
pickle.dump(ret, open(path, 'wb'))
