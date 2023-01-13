import numpy
from numpy import vstack
from pandas import read_csv
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import time
import copy

class ThryoidCSVDataset(Dataset):
    def __init__(self) -> None:
        # read data
        dataURL = 'https://raw.githubusercontent.com/StatsGary/Data/main/thyroid_raw.csv'
        df = pd.read_csv(dataURL, header=None)   
        print(df)

        # modify data
        self.x = df.values[:, :-1]
        self.x = self.x.astype('float32')

        self.y = df.values[:, -1]
        self.y = numpy.array([1 if i == 'sick' else 0 for i in self.y]) # sick = 1, neg = 0
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        print(self.x)
        print(self.y)

        print(self.splitData())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return [self.x[index], self.y[index]]

    def splitData(self, ratio=0.2):
        testSize = round(ratio * len(self.x))
        trainSize = len(self.x) - testSize
        return random_split(self, [testSize, trainSize])

class ThyroidMLP(Module):
    def __init__(self, nInputs) -> None:
        ''' 
        The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.
        https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/#:~:text=The%20rectified%20linear%20activation%20function,otherwise%2C%20it%20will%20output%20zero. 
        '''
        super().__init__()
        self.hidden1 = Linear(nInputs, 20) # output 20 layers
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        self.hidden2 = Linear(20, 10) # 20 layer inputs from hidden 1
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        self.hidden3 = Linear(10, 1) # 10 layer inputs from hidden 2
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid() # allows the classification to fall between 0 – 1, meaning the higher the weight, the more likely the outcome is 1 = sick, than not sick, or negative in the labels case.

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)

        x = self.hidden2(x)
        x = self.act2(x)

        x = self.hidden3(x)
        x = self.act3(x)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
thryoid = ThryoidCSVDataset()