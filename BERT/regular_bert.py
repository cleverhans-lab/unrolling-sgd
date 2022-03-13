import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PyHessian.pyhessian import hessian # Hessian computation
import os
import argparse
import numpy as np
import random 


RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def validate(model,test_dataloader):
    correct = 0
    total = 0
    for images_test, labels_test in test_dataloader:
        images_test =  images_test.to(device)
        labels_test = labels_test.to(device)
        outputs_test = model(images_test)


        _, predicted = torch.max(outputs_test.data, 1)
        total+= labels_test.squeeze().size(0)
        # for gpu, bring the predicted and labels back to cpu fro python operations to work
        correct+= (predicted == labels_test.squeeze()).sum()
    accuracy = 100 * correct.item()/total
    return accuracy

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
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--weight_decay', default=0.0, type=float, help='number of finetuning epochs')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = args.lr
wd = args.weight_decay
std_reg = 0.0
def std_loss(x,y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    avg_std = torch.sum(torch.std(x, dim=1))/(len(x.view(-1)))
    loss = loss + std_reg*avg_std
    l2_norm = 0
    M_weights_tensor = [param for param in model.parameters()]
    curr_weights = weights_to_list_fast(M_weights_tensor)
    l2_norm = np.linalg.norm((np.array(curr_weights)))
    
    loss += l2_norm*wd
    return loss

from torchvision import transforms, datasets
import torch.nn as nn

train_y = torch.load("./imdb_bert_train_labels.pt")
train_x = torch.load("./imdb_bert_train_pooled.pt")

test_y = torch.load("./imdb_bert_test_labels.pt")
test_x = torch.load("./imdb_bert_test_pooled.pt")

class CustomTextDataset(Dataset):
    def __init__(self, txt, labels):
        self.labels = labels
        self.text = txt
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.text[idx]
        sample = (image,label)
        return sample

train_data = CustomTextDataset(train_x,train_y)
test_data= CustomTextDataset(test_x,test_y)

train_dataloader = DataLoader(train_data,batch_size = args.pretrain_batch_size, shuffle = True)
test_dataloader =  DataLoader(test_data,batch_size = 64, shuffle = False)

model = torch.nn.Linear(768, 2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

optimizer = optim.SGD(model.parameters(), lr =lr)

test_acc_list = []

total_len = len(train_dataloader)

for epoch in range(args.pretrain_epochs):
    train_correct = 0
    train_total = 0
    print(f'on pretraining epoch = {epoch}')
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs =model(inputs)

        labels = labels.squeeze()
        loss = std_loss(outputs, labels)
        loss.backward()
        optimizer.step()
    acc = validate(model, test_dataloader)
    test_acc_list.append(acc) 
print(f"Testing accuracy after {args.pretrain_epochs} epoch of pretraining = {acc}")
final_test_acc  = acc

ret = {}
ret['test_acc'] = test_acc_list

import pickle
if not os.path.isdir('regular_results'):
    os.mkdir('regular_results')
path = f'./regular_results/{args.model}_{args.lr}_{args.pretrain_batch_size}_{args.pretrain_epochs}_{std_reg}_{wd}.p'
pickle.dump(ret, open(path, 'wb'))






