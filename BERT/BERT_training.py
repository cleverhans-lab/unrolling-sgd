import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import datasets

from bert_helpers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = datasets.load_dataset("imdb",split = "train")
test_data = datasets.load_dataset("imdb",split = "test")


from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#model = DistilBertModel.from_pretrained("./distilbert-base-uncased")





PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
#PRE_TRAINED_MODEL_NAME = './distilbert-base-uncased/'
#tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BATCH_SIZE = 2
MAX_LEN = 512

train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    #BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    out = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    return self.out(out[1])

model = SentimentClassifier(2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
n_epochs = 1

model = model.to(device)

def eval_model(model, dataloader):
  model.eval()
  total = 0
  correct = 0

  for d in dataloader:
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    targets = data["targets"].to(device)

    out = model(input_ids,attention_mask)
    _, preds = torch.max(out, dim=1)

    correct += torch.sum(preds == targets)
    total += len(targets)

  return (correct,total)


for epoch in range(n_epochs):
  for (i,data) in enumerate(train_data_loader):
    model.train()
    optimizer.zero_grad()
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    targets = data["targets"].to(device)

    out = model(input_ids,attention_mask)
    loss = criterion(out,targets)

    loss.backward()
    optimizer.step()

    print(f"Loss {loss}")

    if i%100 == 0:
      correct,total = eval_model(model,test_data_loader)
      print(f"Train Loss {loss}, Test Accuracy {correct/total}")


