'''
Created on Nov 13, 2018

@author: Yawen
'''
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.autograd import Variable 
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class CSVPathDataset(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv)
    def __len__(self):
        return len(self.df)
    def __getitem__(self):
        y = self.df.tip_amount
        x = self.df.drop('tip_amount', 1)
        x = x.drop('PULocationID', 1)
        x = x.drop('DOLocationID', 1)
        #x = self.df.fare_amount
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        y_train = torch.from_numpy(np.array(y_train))
        y_test = torch.from_numpy(np.array(y_test))
        x_train = torch.from_numpy(np.array(x_train))
        x_test = torch.from_numpy(np.array(x_test))
        return x_train, y_train, x_test, y_test


csvFile = CSVPathDataset("CS516-project/tiny.csv")


x, y, x_test, y_test = csvFile.__getitem__()
#x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

print(x.shape);
print(y.shape)
x = Variable(x.float())
y = Variable(y.float())

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer 1
        #self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer 2
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        out = F.relu(self.hidden1(x))         # activation function for hidden layer
        #out = F.relu(self.hidden2(out))      # activation function for hidden layer
        output = self.predict(out)             # linear output
        return output

net = Net(n_feature=11, n_hidden=15, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


for t in range(200):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

torch.save(net.state_dict(), "CS516-project/trainedModel.pt")

net.eval()
#x_test = x_test.reshape(-1, 1)
x_test = Variable(x_test.float())    
y_predict = net(x_test).data.numpy()
y_test = y_test.numpy()


print(np.column_stack((y_test, y_predict)))
