import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
import torchvision
import torchvision.transforms as transforms
import numpy as np
from ResNet_CIFAR10 import *
from VGG_model import *
import os
import argparse
torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)




parser = argparse.ArgumentParser(description='Finetuning for verification error')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='resnet', type=str, help='resnet or vgg')
parser.add_argument('--dataset', default = 'cifar10', type=str, help ='cifar10, cifar100')
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--regularizer', default=0.0, type=float, help='number of finetuning epochs')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'



Train_counts_path = f"./prs_stats_data/train_counts_{args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}"
Test_counts_path = f"./prs_stats_data/test_counts_{args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}"
Fractions_path = f"./prs_stats_data/fractions_{args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}"

train_probs_path = f"./prs_stats_data/train_probs_{args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}"
val_probs_path = f"./prs_stats_data/val_probs_{args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}"
train_divisions_path = f"./prs_stats_data/train_divisions_{args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}"
val_divisions_path = f"./prs_stats_data/val_divisions_{args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}"

if args.dataset == 'cifar10':
    num_classes = 10
    transform_train = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
if args.dataset == 'cifar100':
    num_classes = 100
    transform_train = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),])
    transform_test = transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),])
    train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    val_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)


if args.model == 'resnet':
    Regular_model = ResNet18(num_classes)
    Shadow_model = ResNet18(num_classes)
if args.model == 'vgg':
    Regular_model = VGG('VGG19',num_classes)
    Shadow_model = VGG('VGG19',num_classes)

Regular_model = Regular_model.to(device)
Shadow_model = Shadow_model.to(device)

if device == 'cuda':
    Regular_model = torch.nn.DataParallel(Regular_model)
    Shadow_model = torch.nn.DataParallel(Shadow_model)
    cudnn.benchmark = True

Regular_model_path = f'./MI_models/Target_model={args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}.pth'
checkpoint = torch.load(Regular_model_path)
Regular_model.load_state_dict(checkpoint['net'])

Shadow_model_path = f'./MI_models/Shadow_model={args.model}_{args.dataset}_epochs={args.pretrain_epochs}_reg={args.regularizer}.pth'
checkpoint = torch.load(Shadow_model_path)
Shadow_model.load_state_dict(checkpoint['net'])

batch_size = 32

train_list = list(train_data)
val_list = list(val_data)
batch_cut_off = int(0.5*len(train_list))
val_cut_off = int(0.5*len(val_data))
train_list_cut = train_list[:batch_cut_off]
val_list_cut = val_list[:val_cut_off]

reg_train_loader = torch.utils.data.DataLoader(
        train_list,
        batch_size, shuffle=False,
        num_workers=2)

reg_val_loader = torch.utils.data.DataLoader(
        val_list,
        batch_size, shuffle=False,
        num_workers=2)

train_loader_updated = torch.utils.data.DataLoader(
        train_list_cut,
        batch_size, shuffle=False,
        num_workers=2)

val_loader_updated = torch.utils.data.DataLoader(
        val_list_cut,
        batch_size=batch_size, shuffle=False,
        num_workers=2)


def safe_log(probs,small_value =1e-30):
  return torch.log(torch.maximum(probs,small_value*torch.ones(probs.shape)))

def Mentr(out,label):
  #print(out*torch.log(torch.ones(out.shape)-out))
  sum = torch.sum(out*safe_log(torch.ones(out.shape)-out),dim=1)
  #print(sum.shape)
  cancel = out[:,label]*safe_log(torch.ones(out[:,label].shape)-out[:,label])
  #print(out[:,label].shape)
  sum = sum - cancel
  #print(sum.shape)


  return -(torch.ones(len(out))-out[:,label])*safe_log(out[:,label]) - sum

def prob_bins(n_bins,outputs,label):
  mentropy_scores = Mentr(outputs,label)
  #print(mentropy_scores.shape)
  range_vals = torch.max(mentropy_scores) - torch.min(mentropy_scores)
  min = torch.min(mentropy_scores)
  divisions = []
  for i in range(1,n_bins):
    divisions.append(min + i*range_vals/n_bins)

  divisions = torch.tensor(divisions)

  probs = torch.zeros(n_bins)
  total = len(outputs)
  for score in mentropy_scores:
    #print(score)
    #print(type(divisions),len(divisions))
    bin = score < divisions
    if torch.sum(bin) != 0:
      non_zero_ind = torch.nonzero(bin).view(-1)[0]
      update_probs = torch.zeros(n_bins)
      update_probs[non_zero_ind] = 1.0
      probs += update_probs/total

    else:
      update_probs = torch.zeros(n_bins)
      update_probs[-1] = 1.0
      probs += update_probs/total

    #print(probs)


  return probs, divisions


outputs_train = [0]
labels_train = [0]

for img, labels in train_loader_updated:
  img = img.cuda()
  labels_train.append(labels)

  out = Shadow_model(img)
  out1 = out.detach()
  outputs_train.append(out1.cpu())

outputs_train = outputs_train[1:]
outputs_train_tens = outputs_train[0]

for i in range(1,len(outputs_train)):
  outputs_train_tens = torch.cat((outputs_train_tens, outputs_train[i]),0)

labels_train = labels_train[1:]
labels_train_tens = labels_train[0]

for i in range(1,len(outputs_train)):
  labels_train_tens = torch.cat((labels_train_tens, labels_train[i]),0)

per_label_outs = []

for i in range(len(outputs_train_tens[0])):
  indices_1 = labels_train_tens == i
  ind_list = torch.nonzero(indices_1).view(-1).tolist()
  labelled_outs = outputs_train_tens[ind_list]
  per_label_outs.append(labelled_outs)

probs_per_label = []
divisions_per_label = []

n_bins = 10

for i in range(len(outputs_train_tens[0])):
  probs,divisions = prob_bins(n_bins,per_label_outs[i],i)
  probs_per_label.append(probs.tolist())
  divisions_per_label.append(divisions.tolist())

probs_per_label_np = np.array(probs_per_label)
divisions_per_label_np = np.array(divisions_per_label)

train_probs = probs_per_label_np
train_divisions = divisions_per_label_np

np.save(train_divisions_path,train_divisions)
np.save(train_probs_path,train_probs)

outputs_train = [0]
labels_train = [0]

for img, labels in val_loader_updated:
  img = img.cuda()
  labels_train.append(labels)

  out = Shadow_model(img)
  out1 = out.detach()
  outputs_train.append(out1.cpu())

outputs_train = outputs_train[1:]
outputs_train_tens = outputs_train[0]

for i in range(1,len(outputs_train)):
  outputs_train_tens = torch.cat((outputs_train_tens, outputs_train[i]),0)

labels_train = labels_train[1:]
labels_train_tens = labels_train[0]

for i in range(1,len(outputs_train)):
  labels_train_tens = torch.cat((labels_train_tens, labels_train[i]),0)

per_label_outs = []

for i in range(len(outputs_train_tens[0])):
  indices_1 = labels_train_tens == i
  ind_list = torch.nonzero(indices_1).view(-1).tolist()
  labelled_outs = outputs_train_tens[ind_list]
  per_label_outs.append(labelled_outs)

probs_per_label = []
divisions_per_label = []

n_bins = 10

for i in range(len(outputs_train_tens[0])):
  probs,divisions = prob_bins(n_bins,per_label_outs[i],i)
  probs_per_label.append(probs.tolist())
  divisions_per_label.append(divisions.tolist())

probs_per_label_np = np.array(probs_per_label)
divisions_per_label_np = np.array(divisions_per_label)

val_probs = probs_per_label_np
val_divisions = divisions_per_label_np


np.save(val_divisions_path,val_divisions)
np.save(val_probs_path,val_probs)

def Get_risks(loader,cut_off):
  outputs_eval = [0]
  labels_eval = [0]

  for i, (img, labels) in enumerate(loader):
    if i < cut_off:
      img = img.cuda()
      labels_eval.append(labels)

      out = model(img)
      out1 = out.detach()
      outputs_eval.append(out1.cpu())

  outputs_eval = outputs_eval[1:]
  outputs_eval_tens = outputs_eval[0]

  for i in range(1,len(outputs_eval)):
    outputs_eval_tens = torch.cat((outputs_eval_tens, outputs_eval[i]),0)

  labels_eval = labels_eval[1:]
  labels_eval_tens = labels_eval[0]

  for i in range(1,len(outputs_eval)):
    labels_eval_tens = torch.cat((labels_eval_tens, labels_eval[i]),0)


  per_label_mentropy_eval = []

  for i in range(len(outputs_eval_tens[0])):
    indices_1 = labels_eval_tens == i
    ind_list = torch.nonzero(indices_1).view(-1).tolist()
    labelled_outs = outputs_eval_tens[ind_list]
    mentropy_outs = Mentr(labelled_outs,i)
    per_label_mentropy_eval.append(mentropy_outs)

  privacy_risk_train = []

  for i in range(len(per_label_mentropy_eval)):
    train_label_probs = torch.from_numpy(train_probs[i])
    train_label_divisions = torch.from_numpy(train_divisions[i])

    test_label_probs = torch.from_numpy(val_probs[i])
    test_label_divisions = torch.from_numpy(val_divisions[i])

    for score in per_label_mentropy_eval[i]:
      train_bin = score < train_label_divisions
      if torch.sum(train_bin) != 0:
        train_ind = torch.nonzero(train_bin).view(-1)[0]

      else:
        train_ind = -1

      test_bin = score < test_label_divisions
      if torch.sum(test_bin) != 0:
        test_ind = torch.nonzero(test_bin).view(-1)[0]

      else:
        test_ind = -1

      risk = train_label_probs[train_ind]/(train_label_probs[train_ind] + test_label_probs[test_ind])

      privacy_risk_train.append(risk)

  return privacy_risk_train

def Get_Counts(train_risks,test_risks):
  train_counts = torch.zeros(10)
  test_counts = torch.zeros(10)
  risk_divisions = torch.arange(0.1,1.1,step=0.1)

  for risk in train_risks:
    pos = risk < risk_divisions
    #print(pos)
    if torch.sum(pos) != 0:
        ind = torch.nonzero(pos).view(-1)[0]

    else:
        ind = -1

    train_counts[ind] += 1

  for risk in test_risks:
    pos = risk < risk_divisions
    if torch.sum(pos) != 0:
        ind = torch.nonzero(pos).view(-1)[0]
        #print("got here")

    else:
        #print("got here 2")
        ind = -1

    test_counts[ind] += 1

  return train_counts,test_counts



model = Regular_model

train_risks = Get_risks(reg_train_loader,100)
test_risks = Get_risks(reg_val_loader,100)

train_counts,test_counts = Get_Counts(train_risks,test_risks)
train_counts_np = train_counts.numpy()
test_counts_np = test_counts.numpy()

np.save(Train_counts_path,train_counts_np)
np.save(Test_counts_path, test_counts_np)


fractions = torch.div(train_counts,train_counts+test_counts)
#nan_inds = torch.nonzero(torch.isnan(fractions))

fractions_np = fractions.numpy()
np.save(Fractions_path,fractions_np)



