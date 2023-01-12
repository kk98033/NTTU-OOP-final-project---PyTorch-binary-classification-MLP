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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataURL = 'https://raw.githubusercontent.com/StatsGary/Data/main/thyroid_raw.csv'


df = pd.read_csv(dataURL, header=None)   
# df.head()
# print(df)
x = df.values[:, :-1]
x = x.astype('float32')

y = df.values[:, -1]
y = numpy.array([1 if i == 'sick' else 0 for i in y])
y = y.astype('float32')
y = y.reshape((len(y), 1))
print(x)
print(y)