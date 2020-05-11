import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def name(self):
        return "MLP"

class MLPNet2(nn.Module):
    def __init__(self):
        super(MLPNet2, self).__init__()
        #self.fc1 = nn.Linear(28*28, 500)
        self.fc1weight = nn.Linear(1, 28*28*500, bias=False)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        one = torch.tensor([1.0])
        if use_cuda:
            one = one.cuda()
        x_fc1_flatten = self.fc1weight(one)
        x_fc1 = x_fc1_flatten.reshape(28*28, 500)
        x2 = F.relu(torch.mm(x, x_fc1))
        #x = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        x4 = self.fc3(x3)
        return x4
    
class MLPNet3(nn.Module):
    def __init__(self):
        super(MLPNet3, self).__init__()
        #self.fc1 = nn.Linear(28*28, 500)
        #self.fc3 = nn.Linear(256, 10)
        """self.fc1weight = nn.Linear(1, 28*28*500, bias=False)
        self.fc2weight = nn.Linear(1, 500*256, bias=False)
        self.fc3weight = nn.Linear(1, 256*10, bias=False)"""
        self.fc1weight = nn.Embedding(1, 28*28*500)
        self.fc2weight = nn.Embedding(1, 500*256)
        self.fc3weight = nn.Embedding(1, 256*10)
        """self.fc1weight = nn.Linear(1, 28*28*500, bias=True)
        self.fc2weight = nn.Linear(1, 500*256, bias=True)
        self.fc3weight = nn.Linear(1, 256*10, bias=True)"""

    def forward(self, x):
        x = x.view(-1, 28*28)
        #one1 = torch.tensor([1.0])
        one1 = torch.tensor([0])
        if use_cuda:
            one1 = one1.cuda()
        #print("self.fc1weight:", [i for i in self.fc1weight.named_parameters()])
        x_fc1_flatten = self.fc1weight(one1)
        #print("x_fc1_flatten:", x_fc1_flatten)
        x_fc1 = x_fc1_flatten.reshape(28*28, 500)
        x2 = F.relu(torch.mm(x, x_fc1))

        one2 = torch.tensor([0])
        if use_cuda:
            one2 = one2.cuda()
        x_fc2_flatten = self.fc2weight(one2)
        x_fc2 = x_fc2_flatten.reshape(500, 256)
        x3 = F.relu(torch.mm(x2, x_fc2))

        one3 = torch.tensor([0])
        if use_cuda:
            one3 = one3.cuda()
        x_fc3_flatten = self.fc3weight(one3)
        x_fc3 = x_fc3_flatten.reshape(256, 10)
        x4 = F.relu(torch.mm(x3, x_fc3))
        return x4
    
## training
model = MLPNet3()
#model = MLPNet2()
#model = MLPNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

torch.save(model.state_dict(), "beginning")
for epoch in range(5):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        #print("ave_loss:", ave_loss)
        #print("loss.data[0]:", loss.data)
        ave_loss = ave_loss * 0.9 + loss.data.item() * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx+1, ave_loss))
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x, requires_grad=False), Variable(target, requires_grad=False)
        out = model(x)
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data.item() * 0.1
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))

torch.save(model.state_dict(), "finished")
  
