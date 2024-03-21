import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from collections import OrderedDict
# Hyper-parameters 
num_epochs = 1
batch_size = 100
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor()])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                    download=True, transform=transform)

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
net = FashionCNN()
torch.save(net.state_dict(), "saved_model/FashionMnist.pt")

def start_training_task():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
    print('Finished Training')
    
    return model.state_dict()

def aggregated_models(client_trainres_dict, n_round):
    # Khởi tạo một OrderedDict để lưu trữ tổng của các tham số của mỗi layer
    sum_state_dict = OrderedDict()

    # Lặp qua các giá trị của dict chính và cộng giá trị của từng tham số vào sum_state_dict
    for client_id, state_dict in client_trainres_dict.items():
        for key, value in state_dict.items():
            if key in sum_state_dict:
                sum_state_dict[key] = sum_state_dict[key] + torch.tensor(value, dtype=torch.float32)
            else:
                sum_state_dict[key] = torch.tensor(value, dtype=torch.float32)

    # Tính trung bình của các tham số
    num_models = len(client_trainres_dict)
    avg_state_dict = OrderedDict((key, value / num_models) for key, value in sum_state_dict.items())
    torch.save(avg_state_dict, f'model_round_{n_round}.pt')
    torch.save(avg_state_dict, "saved_model/FashionMnist.pt")
    #delete parameter in client_trainres to start new round
    client_trainres_dict.clear()
 