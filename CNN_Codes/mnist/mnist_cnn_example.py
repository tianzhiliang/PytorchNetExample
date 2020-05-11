import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

use_cuda = torch.cuda.is_available()

# Step1: Load data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root="data", train=True, transform=trans, download=True)
test_set = dset.MNIST(root="data", train=False, transform=trans, download=True)

batch_size = 100
train_iter = torch.utils.data.DataLoader(
                 dataset=train_set, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(
                dataset=test_set, batch_size=batch_size, shuffle=False)
print('==>>> total trainning batch number: {}'.format(len(train_iter)))
print('==>>> total testing batch number: {}'.format(len(test_iter)))

# Step2: Bulid the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input channel = 1, output channel = 6, kernel_size = 1
        # x: (batchsize, 1, 28, 28) -> (batchsize, 6, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1)
    
        # pool size = 2
        # x: (batchsize, 6, 28, 28) -> (batchsize, 6, 14, 14) 
        self.pool1 = nn.MaxPool2d(2)

        # input channel = 6, output channel = 16, kernel_size = 5
        # x: (batchsize, 6, 14, 14) -> (batchsize, 16, 10, 10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # pool size = 2
        # x: (batchsize, 16, 10, 10) -> (batchsize, 16, 5, 5) 
        self.pool2 = nn.MaxPool2d(2)

        # x: (batchsize, 16*5*5) -> (batchsize, 120)
        self.fc1 = nn.Linear(16*5*5, 120)
        # x: (batchsize, 120) -> (batchsize, 84)
        self.fc2 = nn.Linear(120, 84)
        # x: (batchsize, 84) -> (batchsize, 10) 
        self.fc3 = nn.Linear(84, 10) # class_num = 10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # flatten as one dimension 
        # x: (batchsize, 16, 5, 5) -> (batchsize, 16*5*5)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = CNN()
if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    # Step3: Train the model
    avg_loss = 0
    for batch_index, (x, target) in enumerate(train_iter):
        # Step3.1: numpy data -> pytorch data
        x, target = Variable(x), Variable(target)
        if use_cuda: # cpu -> gpu
            x, target = x.cuda(), target.cuda()

        # Step3.2: Reset
        optimizer.zero_grad()
        # Step3.3: Feed forward
        output = model.forward(x) # Get the outputput of neural network
        loss = criterion(output, target) # Calculate the loss
        avg_loss += loss.item()

        # Step3.4: Back propagation
        loss.backward() # Calculate the gradient of all parameters
        optimizer.step() # Update the parameters by the optimizer

        # Step3.5: Print loss
        if (batch_index+1) % 100 == 0 or (batch_index+1) == len(train_iter):
            print('epoch:{}, index:{}, train loss:{:.6f}'.format(
                epoch, batch_index+1, avg_loss/(batch_index+1)))

    # Step4: Testing
    correct_cnt, total_cnt = 0, 0
    for batch_index, (x, target) in enumerate(test_iter):
        # Step4.1: numpy data -> pytorch data
        x, target = Variable(x), Variable(target)
        if use_cuda:
            x, target = x.cuda(), target.cuda()

        # Step4.2: Feed forward
        output = model(x)
        loss = criterion(output, target)

        # Step4.3: Select the class with maximal probability
        _, pred_label = torch.max(output, 1) 
        total_cnt += x.size()[0] # number of samples in this batch
        correct_cnt += (pred_label == target).sum()
        
        # Step4.4: Get the accuracy on testing set and print
        if(batch_index+1) % 100 == 0 or (batch_index+1) == len(test_iter):
            print('epoch:{}, index:{}, test acc:{:.3f}'.format(
                epoch, batch_index+1, correct_cnt * 1.0 / total_cnt))
