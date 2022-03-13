import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

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

train_dataloader = DataLoader(train_data,batch_size = 64, shuffle = True)
test_dataloader =  DataLoader(test_data,batch_size = 64, shuffle = False)


model = torch.nn.Linear(768, 2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-6)
train_acc_l = []
test_acc_l = []
train_loss_l = []
epoch_list= []


for epoch in range(int(200)):
    train_correct = 0
    train_total = 0
    if epoch%10==0 and epoch !=0:
        print(f"On epoch {epoch} Train Accuracy {train_acc} Test Accuracy {accuracy}")
    for i, (inputs, labels) in enumerate(train_dataloader):

        images = Variable(inputs).to(device)
        labels = Variable(labels).to(device)
        #optimizer.zero_grad()
        outputs = F.sigmoid(model(images))

        _, predicted = torch.max(outputs.data, 1)
        train_total+= labels.squeeze().size(0)
        train_correct+= (predicted == labels.squeeze()).sum()
        labels = labels.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # calculate Accuracy

    correct = 0
    total = 0

    for images_test, labels_test in test_dataloader:
        images_test_var = Variable(images_test).to(device)
        outputs_test = model(images_test_var)
        labels_test_var = labels_test.to(device)
        _, predicted = torch.max(outputs_test.data, 1)
        total+= labels_test.squeeze().size(0)
        # for gpu, bring the predicted and labels back to cpu fro python operations to work
        correct+= (predicted == labels_test_var.squeeze()).sum()
    accuracy = 100 * correct/total
    train_acc = 100 * train_correct/train_total
    #print(f"Epoch: {str(epoch).zfill(2)}. Train Loss: {loss.item()}. Train Accuracy {train_acc}% Test Accuracy: {accuracy}%")
    train_acc_l.append(train_acc)
    test_acc_l.append(accuracy)
    epoch_list.append(epoch)
    train_loss_l.append(loss)
print(f"Maximum test accuracy",max(test_acc_l))
print(f"Maximum test accuracy",max(test_acc_l))
