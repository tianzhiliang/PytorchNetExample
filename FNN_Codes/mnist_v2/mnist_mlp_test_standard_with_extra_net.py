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
        self.with_extra_feature_opt = 8
        # 0: none 1&2: fix random vector 3&4: learnable emb as vector 5&6: learnable param with gate2
        # 1&3&5: only first layer 2&4&6: all layers 

        if self.with_extra_feature_opt in [1, 2]:
            self.rand_input1 = Variable(torch.rand([28*28, 500]), requires_grad=False)
        if self.with_extra_feature_opt in [2]:
            self.rand_input2 = Variable(torch.rand([500, 256]), requires_grad=False)
            self.rand_input3 = Variable(torch.rand([256, 10]), requires_grad=False)

        if self.with_extra_feature_opt in [3, 4]:
            self.emb1 = nn.Embedding(1, 28*28*500)
        if self.with_extra_feature_opt in [4]:
            self.emb2 = nn.Embedding(1, 256*500)
            self.emb3 = nn.Embedding(1, 256*10)
        if self.with_extra_feature_opt in [5,6,7,8]:
            self.emb1 = nn.Embedding(1, 128)
            self.fc_for_weight1 = nn.Linear(128, 28*28*500)
            self.fc_for_gate_fc11 = nn.Linear(128, 128)
            self.fc_for_gate_fc21 = nn.Linear(128, 1)
        if self.with_extra_feature_opt in [6,8]:
            self.emb2 = nn.Embedding(1, 128)
            self.fc_for_weight2 = nn.Linear(128, 256*500)
            self.fc_for_gate_fc12 = nn.Linear(128, 128)
            self.fc_for_gate_fc22 = nn.Linear(128, 1)
            self.emb3 = nn.Embedding(1, 128)
            self.fc_for_weight3 = nn.Linear(128, 256*10)
            self.fc_for_gate_fc13 = nn.Linear(128, 128)
            self.fc_for_gate_fc23 = nn.Linear(128, 1)

        if use_cuda:
            if self.with_extra_feature_opt in [1, 2]:
                self.rand_input1 = self.rand_input1.cuda()
            if self.with_extra_feature_opt in [2]:
                self.rand_input2 = self.rand_input2.cuda()
                self.rand_input3 = self.rand_input3.cuda()
            if self.with_extra_feature_opt in [3, 4]:
                self.emb1 = self.emb1.cuda()
            if self.with_extra_feature_opt in [4]:
                self.emb2 = self.emb2.cuda()
                self.emb3 = self.emb3.cuda()
            if self.with_extra_feature_opt in [5,6,7,8]:
                self.emb1 = self.emb1.cuda()
                self.fc_for_weight1 = self.fc_for_weight1.cuda()
                self.fc_for_gate_fc11 = self.fc_for_gate_fc11.cuda()
                self.fc_for_gate_fc21 = self.fc_for_gate_fc21.cuda()
            if self.with_extra_feature_opt in [6,8]:
                self.emb2 = self.emb2.cuda()
                self.fc_for_weight2 = self.fc_for_weight2.cuda()
                self.fc_for_gate_fc12 = self.fc_for_gate_fc12.cuda()
                self.fc_for_gate_fc22 = self.fc_for_gate_fc22.cuda()
                self.emb3 = self.emb3.cuda()
                self.fc_for_weight3 = self.fc_for_weight3.cuda()
                self.fc_for_gate_fc13 = self.fc_for_gate_fc13.cuda()
                self.fc_for_gate_fc23 = self.fc_for_gate_fc23.cuda()

    def forward(self, x):
        x = x.view(-1, 28*28)

        if self.with_extra_feature_opt in [3, 4, 5, 6, 7, 8]:
            zero = Variable(torch.tensor([0]).long(), requires_grad=False)
            if use_cuda:
                zero = zero.cuda()

        if self.with_extra_feature_opt in [5,6,7,8]:
            emb_vec1 = self.emb1(zero)
            fc1_vec1 = self.fc_for_weight1(emb_vec1)
            if self.with_extra_feature_opt in [5,6]:
                gate1_vec1 = F.relu(self.fc_for_gate_fc11(emb_vec1))
                gate1 = torch.sigmoid(self.fc_for_gate_fc21(gate1_vec1))
                task_vec1 = fc1_vec1 * gate1
            elif self.with_extra_feature_opt in [7,8]:
                task_vec1 = fc1_vec1 

            x = F.relu(self.fc1(x,adapt_vector_for_weight=task_vec1))
        if self.with_extra_feature_opt in [5,7]:
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        elif self.with_extra_feature_opt in [6,8]:
            emb_vec2 = self.emb2(zero)
            fc1_vec2 = self.fc_for_weight2(emb_vec2)
            if self.with_extra_feature_opt in [6]:
                gate1_vec2 = F.relu(self.fc_for_gate_fc12(emb_vec2))
                gate2 = torch.sigmoid(self.fc_for_gate_fc22(gate1_vec2))
                task_vec2 = fc1_vec2 * gate2
            elif self.with_extra_feature_opt in [8]:
                task_vec2 = fc1_vec2
            x = F.relu(self.fc2(x,adapt_vector_for_weight=task_vec2))

            emb_vec3 = self.emb3(zero)
            fc1_vec3 = self.fc_for_weight3(emb_vec3)
            if self.with_extra_feature_opt in [6]:
                gate1_vec3 = F.relu(self.fc_for_gate_fc13(emb_vec3))
                gate3 = torch.sigmoid(self.fc_for_gate_fc23(gate1_vec3))
                task_vec3 = fc1_vec3 * gate3
            elif self.with_extra_feature_opt in [8]:
                task_vec3 = fc1_vec3 

            x = self.fc3(x,adapt_vector_for_weight=task_vec3)
            return x

        if self.with_extra_feature_opt in [1, 2]:
            x = F.relu(self.fc1(x,adapt_vector_for_weight=self.rand_input1))
        elif self.with_extra_feature_opt in [3, 4]:
            x = F.relu(self.fc1(x,adapt_vector_for_weight=self.emb1(zero)))
        else:
            x = F.relu(self.fc1(x))
            
        if self.with_extra_feature_opt in [2]:
            x = F.relu(self.fc2(x,adapt_vector_for_weight=self.rand_input2))
            x = self.fc3(x,adapt_vector_for_weight=self.rand_input3)
        elif self.with_extra_feature_opt in [4]:
            x = F.relu(self.fc2(x,adapt_vector_for_weight=self.emb2(zero)))
            x = self.fc3(x,adapt_vector_for_weight=self.emb3(zero))
        else:
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
