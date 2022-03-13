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
#=====================================================================
#=====================================================================
#=====================================================================


parser = argparse.ArgumentParser(description='Finetuning for verification error')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='resnet', type=str, help='resnet or vgg')
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--regularizer', default=0.0, type=float, help='number of finetuning epochs')
parser.add_argument('--weight_decay', default=0.0, type=float, help='number of finetuning epochs')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = args.lr
wd = args.weight_decay
std_reg =args.regularizer

def std_loss(x,y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    avg_std = torch.sum(torch.std(x, dim=1))/(len(x.view(-1)))
    loss = loss + std_reg*avg_std
    if wd !=0:
        l2_norm = 0
        qq = [param for param in model.parameters()]
        qqq = weights_to_list_fast(qq)
        l2_norm = np.linalg.norm((np.array(qqq)))
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

#loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr =lr)
sigma_list = []
delta_weights_list = []
rolling_unl_error_list = []
unl_error_list = []

M_pretrain = copy.deepcopy(model)
w_pretrain_weights_tensor = [param for param in M_pretrain.parameters()]
w_0_weights = weights_to_list_fast(w_pretrain_weights_tensor)

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
        #Measure unlearning error:
        if i%(total_len-3) == 0 and i !=0:
            print(f"measuring shit...")
            model.eval()
            hessian_comp = hessian(model, std_loss, data=(inputs, labels), cuda=True)
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
            sigma = np.sqrt(top_eigenvalues[-1])
            sigma_list.append(sigma)
            M_weights_tensor = [param for param in model.parameters()]
            curr_weights = weights_to_list_fast(M_weights_tensor)

            delta_weights = np.linalg.norm((np.array(curr_weights) - np.array(w_0_weights)))
            unl_error = (lr * lr) *((len(train_dataloader)*epoch)+ i) *(1/2) * delta_weights *sigma
            rolling_unl_error = (lr * lr) *((len(train_dataloader)*epoch)+ i) *(1/2) * delta_weights * (sum(sigma_list)/len(sigma_list))
            delta_weights_list.append(delta_weights)
            unl_error_list.append(unl_error)
            rolling_unl_error_list.append(rolling_unl_error)
    acc = validate(model, test_dataloader)
    test_acc_list.append(acc)
print(f"Testing accuracy after {args.pretrain_epochs} epoch of pretraining = {acc}")
final_test_acc  = acc

ret = {}
ret['test_acc'] = test_acc_list
ret['sigma'] = sigma_list
ret['delta weights'] = delta_weights_list
ret['unlearning error'] = unl_error_list
ret['rolling unlearning error'] = rolling_unl_error_list


import pickle
if not os.path.isdir('no_bert_results_fixed'):
    os.mkdir('no_bert_results_fixed')
path = f'./no_bert_results_fixed/{args.model}_{args.lr}_{args.pretrain_batch_size}_{args.pretrain_epochs}_{args.regularizer}_{wd}.p'
pickle.dump(ret, open(path, 'wb'))






