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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
thryoid = ThryoidCSVDataset()