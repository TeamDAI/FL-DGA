import numpy as np
import os
import torchvision
from torchvision import datasets, transforms
import random
import torch
import pandas as pd
import torch
import string
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
from tldextract import extract
from datetime import datetime
import time
import os
import psutil
import logging
from lstm import *

r_iid = 0.2
num_clients = 5
epochs=10
lr = 3e-4

LOGGING_DIR = 'D:\\FIL 2024\\testlogs'
LOGGING_FILE = f"logs/app-{datetime.today().strftime('%Y-%m-%d')}.log"

# Create log directory if it doesn't exist
os.makedirs(LOGGING_DIR, exist_ok=True)

logging.basicConfig(filename=LOGGING_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

device = "cpu"

def sampling_nonIID_data(r_iid, num_clients, encoded_domains, padded_domains, encoded_labels):
    alpha = np.ones(num_clients)
    proportions = np.random.dirichlet(alpha)
    print(proportions)

    noniid_domains = []
    noniid_labels = []

    for proportion in proportions:
        num_samples = int(len(encoded_domains) * proportion)
        indices = np.random.choice(len(encoded_domains), num_samples, replace=False)
        noniid_domains.append([padded_domains[i] for i in indices])
        noniid_labels.append([encoded_labels[i] for i in indices])

    # Tạo các tập dữ liệu và DataLoader cho mỗi client
    noniid_datasets = []
    noniid_loaders = []

    for domains, labels in zip(noniid_domains, noniid_labels):
        dataset = TensorDataset(torch.tensor(domains, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        noniid_datasets.append(dataset)
        noniid_loaders.append(loader)
    
    return noniid_domains, noniid_labels

def data_loader():
    data_folder = 'data'
    dga_types = [dga_type for dga_type in os.listdir(data_folder) if os.path.isdir(f"{data_folder}/{dga_type}")]
    print(dga_types)

    my_df = pd.DataFrame(columns=['domain', 'type', 'label'])

    for dga_type in dga_types:
        files = os.listdir(f"{data_folder}/{dga_type}")
        for file in files:
            with open(f"{data_folder}/{dga_type}/{file}", 'r') as fp:
                domains_with_type = [[(line.strip()), dga_type, 1] for line in fp.readlines()]
                appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
                my_df = pd.concat([my_df, appending_df], ignore_index=True)

    with open(f'{data_folder}/benign.txt', 'r') as fp:
        domains_with_type = [[(line.strip()), 'benign', 0] for line in fp.readlines()[:60000]]
        appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
        my_df = pd.concat([my_df, appending_df], ignore_index=True)

    train_test_df, val_df = train_test_split(my_df, test_size=0.1, shuffle=True) 
    print("Train_test_df: \n",train_test_df)
    # Pre-processing
    domains = train_test_df['domain'].to_numpy()
    labels = train_test_df['label'].to_numpy()

    char2ix = {x:idx+1 for idx, x in enumerate([c for c in string.printable])} # alphabelt a:65
    ix2char = {ix:char for char, ix in char2ix.items()}
    print("char2ix: ", char2ix)
    # Convert characters to int and pad
    encoded_domains = [[char2ix[y] for y in x] for x in domains]
    encoded_labels = [0 if x == 0 else 1 for x in labels]
    print("encode domains: ", encoded_domains)
    print("encode label: ", encoded_labels)

    # print(f"Number of samples: {len(encoded_domains)}")
    # print(f"One-hot dims: {len(char2ix) + 1}")

    encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
    encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]
    print("encode domains: ", encoded_domains)
    print("encode label: ", encoded_labels)

    assert len(encoded_domains) == len(encoded_labels)

    padded_domains = pad_sequences(encoded_domains, maxlen)

    print("\nData_split: \n")
    noniid_padded_domain, noniid_encode_labels = sampling_nonIID_data(r_iid, num_clients, encoded_domains, padded_domains, encoded_labels)
    print("nonIID domains: ", noniid_padded_domain)
    print("nonIID labels: ", noniid_encode_labels)

    print(f"Number of samples: {len(encoded_domains[0])}")
    # print(f"One-hot dims: {len(char2ix) + 1}")

    # X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(noniid_padded_domain[0], noniid_encode_labels[0], test_size=0.10, shuffle=True)

    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTMModel(max_features, embed_size, hidden_size, n_layers).to(device)

    # model = BiLSTM(max_features, embed_size, hidden_size, n_layers).to(device)

    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.RMSprop(params=model.parameters(), lr=lr)

    logging.info("Using device: %s", device)

    time_start = time.time()
    logging.info("\n Time start: %d\n", time_start)

    for epoch in range(epochs):
        logging.info("\nEpoch: %d\n", epoch+1)
        train_loss = train(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer, epoch=epoch, batch_size=batch_size)
        eval_loss, accuracy = test(model=model, testloader=testloader, criterion=criterion, batch_size=batch_size)
        print(
            "Epoch: {}/{}".format(epoch+1, epochs),
            "Training Loss: {:.4f}".format(train_loss.item()), 
            "Eval Loss: {:.4f}".format(eval_loss),
            "Accuracy: {:.4f}".format(accuracy)
        )
        ram_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        logging.info("\nEpoch: {}/{} Training Loss: {:.4f} Eval Loss: {:.4f} Accuracy: {:.4f} Ram: {:.4f} CPU: {:.4f}".format(
                        epoch + 1, epochs, train_loss.item(), eval_loss, accuracy, ram_usage, cpu_usage))

    print('Finished Training')
    time_end = time.time()
    logging.info("\n Time end: %d\n", time_end)
    return model.state_dict()

if __name__ == "__main__":
    data_loader()

    