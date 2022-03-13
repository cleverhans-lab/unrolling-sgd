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
parser.add_argument('--epochs', default=60, type=int, help='number of pretraining epochs')
parser.add_argument('--batch_size', default=128, type=int, help='pretraining batch size')
parser.add_argument('--regularizer', default=0.0, type=float, help='number of finetuning epochs')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
std_reg =args.regularizer


train_probs_path = f"./prs_stats_data/train_probs_{args.model}_{args.dataset}_epochs=60_reg={args.regularizer}.npy"
val_probs_path = f"./prs_stats_data/val_probs_{args.model}_{args.dataset}_epochs=60_reg={args.regularizer}.npy"
train_divisions_path =f"./prs_stats_data/train_divisions_{args.model}_{args.dataset}_epochs=60_reg={args.regularizer}.npy"
val_divisions_path = f"./prs_stats_data/val_divisions_{args.model}_{args.dataset}_epochs=60_reg={args.regularizer}.npy"


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
if args.model == 'resnet50':
    net = ResNet50(num_classes)
if args.model == 'resnet101':
    net = ResNet101(num_classes)
if args.model == 'vgg':
    net = VGG('VGG19',num_classes)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

trainset_list = list(trainset)
batch_star = trainset_list[-args.batch_size:]
data_no_unlearned =  trainset_list[:-args.batch_size]

#the data that we are finetuning on (with x* at the beginning)
data_loader_minus_x = torch.utils.data.DataLoader(data_no_unlearned, batch_size = args.batch_size, shuffle= False, num_workers =2)
#The unlearned loader
unlearned_loader = torch.utils.data.DataLoader(batch_star, batch_size = args.batch_size, shuffle=False, num_workers=2)

#Entirety of D
data_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle= False, num_workers =2)

#Test dataset 
test_loader = torch.utils.data.DataLoader(testset,args.batch_size,shuffle = False)

print(f'The following should be true ...{len(unlearned_loader)} + {len(data_loader_minus_x)} = {len(data_loader)}')


#step 1: pretrain for a very small number of steps on D-x
pretrain_optimizer = optim.SGD(net.parameters(), lr = 0.01,momentum=0.9, weight_decay=5e-4)
for ep in range(0,5):
  print(f'Pretrain Epoch = {ep}')
  for i,(imgs,labels) in enumerate(data_loader_minus_x):
    imgs = imgs.to(device)
    labels = labels.to(device)
    pretrain_optimizer.zero_grad()
    out = net(imgs)
    loss = std_loss(out,labels)
    loss.backward()
    pretrain_optimizer.step()

correct, total = validate(net, test_loader)
print(f"Testing accuracy after {args.epochs} epoch of pretraining = {100*correct/total}")

#step 2: Get the gradient we need to calculate M''
net.train()
for i,(img,label) in enumerate(unlearned_loader):
    img = img.cuda()
    label = label.cuda()
    output_pre = net(img)
    loss_unl = std_loss(output_pre,label)
    loss_unl.backward(retain_graph=True)
    grads = torch.autograd.grad(loss_unl, [param for param in net.parameters()],create_graph = True)
grads_list = []
#step 3: Let's now train M and M' for many epochs and at the very end, let's get M'' and calculate PRS at the very end

#Need to save the weights at the beginning
M_pretrain = copy.deepcopy(net)
w_pretrain_weights_tensor = [param for param in M_pretrain.parameters()]
w_pretrain_weights = weights_to_list_fast(w_pretrain_weights_tensor)

M = copy.deepcopy(net)
M_retrained = copy.deepcopy(net)

M.train()
M_retrained.train()

regular_optim = optim.SGD(M.parameters(), lr = 0.01,momentum=0.9, weight_decay=5e-4)
retrain_optim = optim.SGD(M_retrained.parameters(), lr = 0.01,momentum=0.9, weight_decay=5e-4)


for ep in range(0,args.epochs):
     print(f'PRS Epoch = {ep}')
     for i,(imgs,labels) in enumerate(data_loader):
         #update M:
         imgs = imgs.to(device)
         labels = labels.to(device)
         regular_optim.zero_grad()
         out =  M(imgs)
         loss = std_loss(out,labels)
         loss.backward()
         regular_optim.step()

         #Update M'
         if i!=0:
             retrain_optim.zero_grad()
             out = M_retrained(imgs)
             loss = std_loss(out,labels)
             loss.backward()
             retrain_optim.step()
         if i==0:
             M.train()
             for j,(img,label) in enumerate(unlearned_loader):
                img = img.cuda()
                label = label.cuda()
                output_pre = M(img)
                loss_unl = std_loss(output_pre,label)
                #loss_unl.backward(retain_graph = True)
                loss_unl.backward(retain_graph = True)
                grads =  torch.autograd.grad(loss_unl, [param for param in M.parameters()])
                #grads = torch.autograd.grad(loss_unl, [param for param in M.parameters()],retain_graph = False,create_graph = True)
             #grads = torch.tensor(grads)
             #grads = grads.detach().cpu()
             grads_list.append(grads)
#Now, save the weights of both M_(N+t) and M'_(N+t)
M_weights_tensor = [param for param in M.parameters()]
w_M_weights = weights_to_list_fast(M_weights_tensor)
M_retrain_tensor = [param for param in M_retrained.parameters()]
w_M_retrain_weights = weights_to_list_fast(M_retrain_tensor)


##Now let's get M''
M_unlearned = copy.deepcopy(M)
optimizer_unlearned = optim.SGD(M_unlearned.parameters(), lr = 0.01,momentum=0.9, weight_decay=5e-4)

M_unlearned.train()
old_params = {}
for i, (name, params) in enumerate(M.named_parameters()):
    old_params[name] = params.clone()
    for grads in grads_list:
        old_params[name] += 0.01 * grads[i]
for name, params in M_unlearned.named_parameters():
    params.data.copy_(old_params[name])
M_unlearned_tensor = [param for param in M_unlearned.parameters()]
w_M_unlearned_weights = weights_to_list_fast(M_unlearned_tensor)

verification_error = np.linalg.norm((np.array(w_M_retrain_weights) - np.array(w_M_unlearned_weights)))

privacy_risk_M = Get_risks(unlearned_loader,args.batch_size,M,train_probs_path,val_probs_path,train_divisions_path,val_divisions_path)
privacy_risk_M = torch.tensor(privacy_risk_M)

privacy_risk_unlearned = Get_risks(unlearned_loader, args.batch_size, M_unlearned,train_probs_path,val_probs_path,train_divisions_path,val_divisions_path)
privacy_risk_unlearned = torch.tensor(privacy_risk_unlearned)

print(type(privacy_risk_M))
print(f"finally done, prs was {privacy_risk_M.mean().item()}, now saving files")
ret = {}
ret['verification error'] = verification_error
ret['privacy risk score of M'] = privacy_risk_M.mean().item()
ret['privacy risk score of M unlearned'] = privacy_risk_unlearned.mean().item()
import pickle
if not os.path.isdir('amnesiac_MI_PRS_CORR_results'):
    os.mkdir('amnesiac_MI_PRS_CORR_results')
path = f'./amnesiac_MI_PRS_CORR_results/{args.model}_{args.dataset}_{args.epochs}_{args.regularizer}.p'
pickle.dump(ret, open(path, 'wb'))






