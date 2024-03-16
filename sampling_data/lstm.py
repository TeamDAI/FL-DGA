import numpy as np
import torch
import json
import string
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

from tldextract import extract

device = "cpu"
# Load model
# This should be set in a config file
max_features = 101 # max_features = number of one-hot dimensions
maxlen = 127
embed_size = 64
hidden_size = 64
n_layers = 1
batch_size = 64
char2ix = {x:idx+1 for idx, x in enumerate([c for c in string.printable])}
ix2char = {ix:char for char, ix in char2ix.items()}

class LSTMModel(nn.Module):
    def __init__(self, feat_size, embed_size, hidden_size, n_layers):
        super(LSTMModel, self).__init__()
        
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(feat_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x, hidden):
        embedded_feats = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded_feats, hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        fc_out = self.fc(lstm_out)
        sigmoid_out = self.sigmoid(fc_out)
        sigmoid_out = sigmoid_out.view(x.shape[0], -1)
        sigmoid_last = sigmoid_out[:,-1]

        return sigmoid_last, hidden
    
    def init_hidden(self, x):
        weight = next(self.parameters()).data
        h = (weight.new(self.n_layers, x.shape[0], self.hidden_size).zero_(),
             weight.new(self.n_layers, x.shape[0], self.hidden_size).zero_())
        return h
    
    def get_embeddings(self, x):
        return self.embedding(x)

# Create a bidirectional LSTM model class
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0]*(maxlen-len(domain))+domain)
    return np.asarray(domains)

def evaluate(model, testloader, batch_size):
    y_pred = []
    y_true = []

    h = model.init_hidden(batch_size)
    model.eval()
    for inp, lab in testloader:
        h = tuple([each.data for each in h])
        out, h = model(inp, h)
        y_true.extend(lab)
        preds = torch.round(out.squeeze())
        y_pred.extend(preds)

    print(roc_auc_score(y_true, y_pred))
    
def decision(x):
    return x >= 0.5

def load_data(df):
    """
        Input pandas DataFrame
        Output DataLoader
    """
    max_features = 101 # max_features = number of one-hot dimensions
    maxlen = 127
    batch_size = 64

    domains = df['domain'].to_numpy()
    labels = df['label'].to_numpy()

    char2ix = {x:idx+1 for idx, x in enumerate([c for c in string.printable])}
    ix2char = {ix:char for char, ix in char2ix.items()}

    # Convert characters to int and pad
    encoded_domains = [[char2ix[y] for y in x] for x in domains]
    encoded_labels = [0 if x == 0 else 1 for x in labels]
    encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
    encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

    assert len(encoded_domains) == len(encoded_labels)

    padded_domains = pad_sequences(encoded_domains, maxlen)
    trainset = TensorDataset(torch.tensor(padded_domains, dtype=torch.long), torch.Tensor(encoded_labels))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    return trainloader

def domain2tensor(domains):
    encoded_domains = [[char2ix[y] for y in domain] for domain in domains]
    padded_domains = pad_sequences(encoded_domains, maxlen)
    tensor_domains = torch.LongTensor(padded_domains)
    return tensor_domains

def train(model, trainloader, criterion, optimizer, epoch, batch_size):
    model.train()
    clip = 5
    h = model.init_hidden(domain2tensor(["0"]*batch_size))
    for inputs, labels in (tqdm(trainloader)):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        h = tuple([each.data for each in h])

        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return loss

def test(model, testloader, criterion, batch_size):
    val_h = model.init_hidden(domain2tensor(["0"]*batch_size))
    model.eval()
    eval_losses= []
    total = 0
    correct = 0
    for eval_inputs, eval_labels in testloader:
        
        eval_inputs = eval_inputs.to(device)
        eval_labels = eval_labels.to(device)
        
        val_h = tuple([x.data for x in val_h])
        eval_output, val_h = model(eval_inputs, val_h)
        
        eval_prediction = decision(eval_output)
        total += len(eval_prediction)
        correct += sum(eval_prediction == eval_labels)
        
        eval_loss = criterion(eval_output.squeeze(), eval_labels.float())
        eval_losses.append(eval_loss.item())

    return np.mean(eval_losses), correct/total

def save_state_dict(model, path):
    # print(model.state_dict().items())
    with open(path, 'w') as fp:
        json.dump(fp=fp, obj={k:v.cpu().numpy().tolist() for k,v in model.state_dict().items()})

def load_state_dict(model, path):
    # Need to initialize a new similar model and then apply loaded state_dict
    with open(path, 'r') as fp:
        state_dict = json.load(fp=fp)
        state_dict = {k:torch.tensor(np.array(v)).to(device=device) for k,v in state_dict.items()}
        model.load_state_dict(state_dict)