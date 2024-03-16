import numpy as np
import torchvision
from torchvision import datasets, transforms
import random
import torch
import pandas as pd
import torch
import json
import string
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
from tldextract import extract
from datetime import datetime

num_epochs = 1
batch_size = 100
learning_rate = 0.001

# transform = transforms.Compose(
#     [transforms.ToTensor()])

# train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
#                                         download=True, transform=transform)

# test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
#                                     download=True, transform=transform)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                         shuffle=True)

# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                         shuffle=False)

# k_list = [5, 10, 8] # có nghĩa là người dùng đầu tiên sẽ có 5 mẫu từ mỗi lớp, người dùng thứ hai sẽ có 10 mẫu từ mỗi lớp, và người dùng thứ ba sẽ có 8 mẫu từ mỗi lớp.

# n_list = [3, 5, 2] #có nghĩa là người dùng đầu tiên sẽ có 3 lớp, người dùng thứ hai sẽ có 5 lớp, và người dùng thứ ba sẽ có 2 lớp.
# num_classes = 10

# def fashionmnist_iid(dataset, num_users):
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

# def fashionmnist_non_iid(dataset, num_users):
#     num_shards, num_imgs = 10, 6000
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#     label_begin = {}
#     cnt=0
#     for i in idxs_labels[1,:]:
#         if i not in label_begin:
#                 label_begin[i] = cnt
#         cnt+=1

#     classes_list = []
#     for i in range(num_users):
#         n = n_list[i]
#         k = k_list[i]
#         # k_len = args.train_shots_max
#         k_len = 10
#         classes = random.sample(range(0,num_classes), n)
#         classes = np.sort(classes)
#         print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
#         print("classes:", classes)
#         user_data = np.array([])
#         for each_class in classes:
#             # begin = i*10 + label_begin[each_class.item()]
#             begin = i * k_len + label_begin[each_class.item()]
#             user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
#         dict_users[i] = user_data
#         classes_list.append(classes)

#     return dict_users, classes_list

# # DGA - iid dataset

# x,y = fashionmnist_non_iid(train_dataset, 20)
# print("dict_user: ", x)
# print("class_list: ", y)













def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0]*(maxlen-len(domain))+domain)
    return np.asarray(domains)

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

