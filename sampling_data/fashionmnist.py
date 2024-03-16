import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

import logging
import psutil
from datetime import datetime
import os
import time

LOGGING_DIR = 'D:\\FIL 2024\\test\\logs'
LOGGING_FILE = f"logs/app-{datetime.today().strftime('%Y-%m-%d')}.log"

# Create log directory if it doesn't exist
os.makedirs(LOGGING_DIR, exist_ok=True)

logging.basicConfig(filename=LOGGING_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

num_epochs = 2
batch_size = 100
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor()])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=False, transform=transform)

test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                    download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False)

class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=32*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim=1)
        return out


def start_training_task():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    time_start = time.time()
    logging.info("\n Time start: %d\n", time_start)
    # logging.info("\nEpoch: %d\n", num_epochs)
    print("Using device:", device)

    model = FashionCNN().to(device)
    #model.load_state_dict(torch.load('newmode.pt'))
    model.load_state_dict(torch.load('newmode.pt', map_location=device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    total = 0
    correct = 0
    for epoch in range(num_epochs):
        print("\n","Epoch: ", epoch , "\n")
        logging.info("\nEpoch: %d\n", epoch)
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            #if (i+1) % 600 == 0:
                #print (f' Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            ram_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            logging.info("Epoch [%d/%d], Step [%d/%d], Loss: %.4f, RAM Usage: %.2f%%, CPU Usage: %.2f%%",
                         epoch+1, num_epochs, i+1, n_total_steps, loss.item(), ram_usage, cpu_usage)
            
        # logging.info(f" Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss}")
        # logging.info(f"Total Ram: [{ram_usage}] | CPU: [{cpu_usage}]")
        print(f" Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss:", loss)

    print('Finished Training')
    time_end = time.time()
    logging.info("\n Time start: %d\n", time_end)
    return model.state_dict()

start_training_task()