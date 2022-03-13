import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy
import numpy as np


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

def Get_risks(loader,cut_off,model,a,b,c,d):
  outputs_eval = [0]
  labels_eval = [0]
  train_probs = np.load(a)
  val_probs = np.load(b)

  train_divisions = np.load(c)
  val_divisions = np.load(d)
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
  risk_divisions = torch.arange(0,1.1,step=0.1)
  train_probs = np.load('./train_probs.npy')
  val_probs = np.load('./val_probs.npy')

  train_divisions = np.load('./train_divisions.npy')
  val_divisions = np.load('./val_divisions.npy')
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

    else:
        ind = -1

    test_counts[ind] += 1

    return train_counts,test_counts




