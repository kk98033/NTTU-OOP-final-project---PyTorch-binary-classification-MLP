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
import math

''' temp '''
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score

class ThryoidCSVDataset(Dataset):
    def __init__(self, dataURL) -> None:
        # read data
        # dataURL = 'https://raw.githubusercontent.com/StatsGary/Data/main/thyroid_raw.csv'
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

def trainModel(trainDL, model, epochs=100, lr=0.01, momentum=0.9, savedPath='model.pth'):
    start = time.time()
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = 0.0

    for epoch in range(epochs):
        print(f'Epoch { epoch+1 }/{ epochs }')
        model.train()

        # iterate training data loader
        for i, (inputs, targets) in enumerate(trainDL):
            optimizer.zero_grad() # set to zero gradients
            outputs = model(inputs)
            temp, preds = torch.max(outputs.data, 1) # get class labels
            loss = criterion(outputs, targets)
            loss.backward() # set the loss to back propagate through the network updating the weights as it goes
            optimizer.step()
        torch.save(model, savedPath)

    timeDelta = time.time() - start
    print(f'Trainging complete in { timeDelta // 60 }, { timeDelta % 60 }s')

    return model

def evaluateModel(testDL, model, beta=1.0):
    preds = []
    actuals = []

    for (i, (inputs, targets)) in enumerate(testDL):
        yhat = model(inputs) # evaluate the model on teh test set
        yhat = yhat.detach().numpy()

        actual = targets.numpy() # extract the weights to ndarray instead of tensor
        actual = actual.reshape((len(actual), 1))

        yhat = yhat.round()

        preds.append(yhat)
        actuals.append(actual)
    
    # TODO: confusion matrix
    preds, actuals = vstack(preds), vstack(actuals)
    cm = confusion_matrix(actuals, preds)
    tn, fp, fn, tp = cm.ravel()
    total = sum(cm.ravel())

    metrics = {
        'accuracy': accuracy_score(actuals, preds),
        'AU_ROC': roc_auc_score(actuals, preds),
        'f1_score': f1_score(actuals, preds),
        'average_precision_score': average_precision_score(actuals, preds),
        'f_beta': ((1+beta**2) * precision_score(actuals, preds) * recall_score(actuals, preds)) / (beta**2 * precision_score(actuals, preds) + recall_score(actuals, preds)),
        'matthews_correlation_coefficient': (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        'precision': precision_score(actuals, preds),
        'recall': recall_score(actuals, preds),
        'true_positive_rate_TPR':recall_score(actuals, preds),
        'false_positive_rate_FPR':fp / (fp + tn) ,
        'false_discovery_rate': fp / (fp +tp),
        'false_negative_rate': fn / (fn + tp) ,
        'negative_predictive_value': tn / (tn+fn),
        'misclassification_error_rate': (fp+fn)/total ,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        #'confusion_matrix': confusion_matrix(actuals, preds), 
        'TP': tp,
        'FP': fp, 
        'FN': fn, 
        'TN': tn
    }
    return metrics, preds, actuals

def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy() # get numpy array
    return yhat   

def prepareDataset(path):
    dataset = ThryoidCSVDataset(path)
    train, test = dataset.splitData(ratio=0.1)

    # data loaders
    trainDL = DataLoader(train, batch_size=32, shuffle=True)
    testDL = DataLoader(test, batch_size=1024, shuffle=False)

    return trainDL, testDL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainDL, testDL = prepareDataset('https://raw.githubusercontent.com/StatsGary/Data/main/thyroid_raw.csv')
model = ThyroidMLP(26)
trainModel(trainDL=trainDL, model=model, epochs=150, lr=0.01)

results = evaluateModel(testDL, model, beta=1)
modelMetrics = results[0]
metricsDF = pd.DataFrame.from_dict(modelMetrics, orient='index', columns=['metric'])
metricsDF.reset_index(inplace=True)
metricsDF.to_csv('confusion_matrix_thyroid.csv', index=False)