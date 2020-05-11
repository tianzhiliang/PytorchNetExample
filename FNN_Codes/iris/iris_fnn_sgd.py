import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Check the availability of cuda (GPU)
use_cuda = torch.cuda.is_available() 

# Step1: Load data and preprocessing
# Step1.1: Load
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"
train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
    dtype=float, delimiter=',') 
test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
    dtype=float, delimiter=',') 

# Step1.2: Split x and y (feature and target)
xtrain = train_data[:,:4] # x: 1~4 columns 
ytrain = train_data[:,4] # y: 5-th column

xtest = test_data[:,:4] # x: 1~4 columns
ytest = test_data[:,4] # y: 5-th column

"""
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 32 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris
"""
# Step2: Build model
# Step2.1: Define hyper-parameters
input_feature_dim = 4 # dim of x
hidden_dim = 32 # determined by users
class_num = 3 # candidate set (0,1,2) of y
learing_rate = 0.01
num_epoch = 1000

# Step2.2: build neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_feature_dim, hidden_dim) # W*x+b
        self.fc2 = nn.Linear(hidden_dim, class_num)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output
        
net = Net()
if use_cuda:
    net = net.cuda() # model on cpu -> model on gpu 

# Step2.3: Choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learing_rate)

# Step3: Train the model
for epoch in range(num_epoch):
    # Step3.1: numpy data -> pytorch data
    X = torch.Tensor(xtrain).float()# X (|training_samples| * input_feature_dim)
    Y = torch.Tensor(ytrain).long()# Y (|training_samples| * 1)
    if use_cuda: # data on cpu -> data on gpu
        X = X.cuda()
        Y = Y.cuda()

    # Step3.2: Reset
    optimizer.zero_grad()

    # Step3.3: Feed forward
    output = net.forward(X) # Get the output of neural network 
    # net.forward(X) can also be written as net(X) for short
    loss = criterion(output, Y) # Calculate the loss

    # Step3.4: Back propagation
    loss.backward() # Calculate the gradient of all parameters
    optimizer.step() # Update the parameters by the optimizer

    # Step3.5: Get the accuracy on this epoch and print
    acc = 100 * torch.sum(Y==torch.max(output.data, 1)[1]).double() / len(Y)
    print ('Epoch [%d/%d] Loss: %.4f   Accuracy: %.4f' 
                   %(epoch+1, num_epoch, loss.item(), acc.item()))

# Step4: Test the model
# Step4.1: Get prediction data (numpy data -> pytorch data)
X = torch.Tensor(xtest).float() # X (|testing_samples| * input_feature_dim)
Y = torch.Tensor(ytest).long() # Y (|testing_samples| * 1)
if use_cuda: # data on cpu -> data on gpu
    X = X.cuda()
    Y = Y.cuda()

# Step4.2: Feed forward
output = net(X)

# Step4.3: Select the class with maximal probability
_, predicted = torch.max(output.data, 1) 

# Step4.4: Get the accuracy on testing set and print
print('Accuracy of testing %.4f %%' % (100 * torch.sum(Y==predicted).double() / len(Y)))
