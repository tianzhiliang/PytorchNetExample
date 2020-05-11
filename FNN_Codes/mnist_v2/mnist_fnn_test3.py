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
 
def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)

def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def kaiming_uniform(tensor, nonlinearity='leaky_relu', use_dim0 = True, a=0):
    if use_dim0:
        dim = tensor.shape[0]
    else:
        dim = tensor.shape[1]
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(dim)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

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
class FnnNet(nn.Module):
    def __init__(self):
        super(FnnNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 10, bias=False)
        #self.fc1 = nn.Linear(28*28, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        #print("self.fc1:", [i for i in self.fc1.named_parameters()])
        x = F.relu(self.fc1(x))
        return x
    
    def name(self):
        return "FNN"

class FnnNet2(nn.Module):
    def __init__(self):
        super(FnnNet2, self).__init__()
        #self.fc1 = nn.Linear(28*28, 10)
        #self.fc1weight = nn.Linear(1, 28*28*10, bias=False)
        self.fc1weight = nn.Embedding(1, 28*28*10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        #x = F.relu(self.fc1(x))
        #one1 = torch.tensor([1.0])
        one1 = torch.tensor([0])
        if use_cuda:
            one1 = one1.cuda()
        torch.set_printoptions(profile="full")
        print("self.fc1weight:", [i for i in self.fc1weight.named_parameters()])
        x_fc1_flatten = self.fc1weight(one1)
        print("x_fc1_flatten:", x_fc1_flatten)
        x_fc1 = x_fc1_flatten.reshape(28*28, 10)
        print("x_fc1:", x_fc1)
        print("x:", x)
        x2 = torch.mm(x, x_fc1)
        print("x2:", x2)
        torch.set_printoptions(profile="default")
        x3 = F.relu(x2)
        return x3

class FnnNet3(nn.Module):
    def __init__(self):
        super(FnnNet3, self).__init__()
        #self.fc1weight = nn.Linear(1, 28*28*10, bias=False)
        #self.fc1weight = nn.Linear(1, 28*28*10, bias=False)
        #self.fc1weight = nn.Embedding(1, 28*28*10)
        self.fc1weight = nn.Embedding(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        #x = F.relu(self.fc1(x))
        one1 = torch.tensor([i for i in range(28*28)])
        #one1 = torch.tensor([1.0])
        #one1 = torch.tensor([0])
        if use_cuda:
            one1 = one1.cuda()
        #print("self.fc1weight:", [i for i in self.fc1weight.named_parameters()])
        #x_fc1_flatten = self.fc1weight(one1)
        #print("x_fc1_flatten:", x_fc1_flatten)
        #x_fc1 = x_fc1_flatten.reshape(28*28, 10)
        x_fc1 = self.fc1weight(one1)
        x2 = F.relu(torch.mm(x, x_fc1))
        sys.stdout.flush()
        return x2

class FnnNet4(nn.Module):
    def __init__(self):
        super(FnnNet4, self).__init__()
        #self.fc1 = nn.Linear(28*28, 10)
        self.use_bias = True
        self.fc1weight = nn.Linear(1, 28*28*10, bias=False)
        if self.use_bias:
            self.fc1bias = nn.Linear(1, 10, bias=False)
        #self.fc1weight = nn.Embedding(1, 28*28*10)
        #print("0 self.fc1weight:", [i for i in self.fc1weight.named_parameters()])
        for p in self.fc1weight.parameters():
            init.kaiming_uniform_(p, a=math.sqrt(5))
        if self.use_bias:
            #for p in self.fc1bias.parameters():
            #    init.kaiming_uniform_(p, a=math.sqrt(5))
            fc1weight_param = [p for p in self.fc1weight.parameters()][0]
            fan_in, _ = init._calculate_fan_in_and_fan_out(fc1weight_param)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(p, -bound, bound)
        #print("1 self.fc1weight:", [i for i in self.fc1weight.named_parameters()])

    def forward(self, x):
        x = x.view(-1, 28*28)
        #x = F.relu(self.fc1(x))
        #one1 = torch.tensor([0])
        one1 = torch.tensor([1.0])
        if use_cuda:
            one1 = one1.cuda()
        torch.set_printoptions(profile="full")
        #print("self.fc1weight:", [i for i in self.fc1weight.named_parameters()])
        x_fc1_flatten = self.fc1weight(one1)
        #print("x_fc1_flatten:", x_fc1_flatten)
        #x_fc1 = x_fc1_flatten.reshape(28*28, 10)
        x_fc1 = x_fc1_flatten.reshape(10, 28*28)

        if self.use_bias:
            one2 = torch.tensor([1.0])
            if use_cuda:
                one2 = one2.cuda()
            x_fc1bias_flatten = self.fc1bias(one2)
            x_fc1bias = x_fc1bias_flatten.reshape(10)
            x2 = F.linear(x, x_fc1, x_fc1bias)
        else:
            x2 = F.linear(x, x_fc1, None)
        #print("x_fc1:", x_fc1)
        #print("x:", x)
        #x2 = F.linear(x, x_fc1, None)
        #x22 = torch.mm(x, x_fc1.t())
        #print("x2:", x2)
        #print("x22:", x22)
        torch.set_printoptions(profile="default")
        x3 = F.relu(x2)
        return x3

class FnnNet5(nn.Module):
    def __init__(self):
        super(FnnNet5, self).__init__()
        #self.x_fc1 = nn.Parameter(torch.rand(28*28,10))
        self.x_fc1 = nn.Parameter(torch.rand(28*28,10))
        kaiming_uniform(self.x_fc1)
        #print("0 self.fc1weight:", [i for i in self.fc1weight.named_parameters()])
        #for p in self.fc1weight.parameters():
        #    init.kaiming_uniform_(p, a=math.sqrt(5))
        """if self.use_bias:
            #for p in self.fc1bias.parameters():
            #    init.kaiming_uniform_(p, a=math.sqrt(5))
            fc1weight_param = [p for p in self.fc1weight.parameters()][0]
            fan_in, _ = init._calculate_fan_in_and_fan_out(fc1weight_param)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(p, -bound, bound)"""
        #print("1 self.fc1weight:", [i for i in self.fc1weight.named_parameters()])

    def forward(self, x):
        x = x.view(-1, 28*28)
        #torch.set_printoptions(profile="full")
        #print("x_fc1:", x_fc1)
        #print("x:", x)
        x2 = torch.mm(x, self.x_fc1)
        #print("x2:", x2)
        #torch.set_printoptions(profile="default")
        x3 = F.relu(x2)
        return x3

class FnnNet6(nn.Module):
    def __init__(self):
        super(FnnNet6, self).__init__()
        self.x_fc1 = nn.Parameter(torch.rand(28*28,10))
        kaiming_uniform(self.x_fc1)
        #print("0 self.fc1weight:", [i for i in self.fc1weight.named_parameters()])
        """if self.use_bias:
            fc1weight_param = [p for p in self.fc1weight.parameters()][0]
            fan_in, _ = init._calculate_fan_in_and_fan_out(fc1weight_param)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(p, -bound, bound)"""
        #print("1 self.fc1weight:", [i for i in self.fc1weight.named_parameters()])

    def forward(self, x):
        x = x.view(-1, 28*28)
        #torch.set_printoptions(profile="full")
        #print("x_fc1:", x_fc1)
        #print("x:", x)
        x2 = torch.mm(x, self.x_fc1)
        #print("x2:", x2)
        #torch.set_printoptions(profile="default")
        x3 = F.relu(x2)
        return x3

## training
#model = FnnNet()
#model = FnnNet2()
#model = FnnNet3()
#model = FnnNet4()
#model = FnnNet5()
model = FnnNet6()

if use_cuda:
    model = model.cuda()

lr = 0.01
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=lr)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

torch.save(model.state_dict(), "beginning_new")
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
        sys.stdout.flush()
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

torch.save(model.state_dict(), "finished_new")
  
