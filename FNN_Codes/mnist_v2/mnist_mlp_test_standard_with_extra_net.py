import os,sys,math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.init as init
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

class Linear_with_param_adapt(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super(Linear_with_param_adapt, self).__init__()
        self.use_bias = bias
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = nn.Parameter(torch.rand(self.dim_out, self.dim_in))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.use_bias:
            self.bias = nn.Parameter(torch.rand(self.dim_out))
            bound = 1 / math.sqrt(self.dim_out)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adapt_vector_for_weight=None, adapt_vector_for_bias=None):
        if adapt_vector_for_weight is not None:
            adapt = adapt_vector_for_weight.reshape(self.weight.shape)
            weight = self.weight * adapt
        else:
            weight = self.weight

        if (adapt_vector_for_bias is not None) and self.use_bias:
           adapt = adapt_vector_for_bias.reshape(self.bias.shape)
           bias = self.bias * adapt
        else:
           bias = self.bias

        code_verion = 1 # both 0 and 1 are OK
        if code_verion == 0:
            W_times_x = torch.mm(x, weight.t())
            if self.use_bias:
                b = bias.repeat(W_times_x.shape[0], 1)
                y = W_times_x + b
            else:
                y = W_times_x
        elif code_verion == 1:
            y = F.linear(x, weight, bias)
        return y

class MLPNet2(nn.Module):
    def __init__(self):
        super(MLPNet2, self).__init__()
        self.fc1 = Linear_with_param_adapt(28*28, 500)
        self.fc2 = Linear_with_param_adapt(500, 256)
        self.fc3 = Linear_with_param_adapt(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def name(self):
        return "MLP2"

def main():
    ## training
    model = MLPNet2()
    #model = MLPNet()

    if use_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    #torch.save(model.state_dict(), "beginning")
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

    #torch.save(model.state_dict(), "finished")

main()
