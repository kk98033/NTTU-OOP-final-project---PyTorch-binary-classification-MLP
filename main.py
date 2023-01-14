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
import matplotlib.pyplot as plt

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
    '''
    trainDL            – dataloader
    epochs             – how many times do we want to feed through and back propagate errors (this is an optional parameter as the default is set to 100)
    lr (learning rate) – at what rate the model learns at – too high a value and it misses key updates, too low and it can take forever. This is where the art comes into Machine Learning.
    save_path          – this will be the serialised PyTorch (.pth) file format. All PyTorch models are saved with this postfix.
    momentum           – is used to speed up training

    https://stackoverflow.com/questions/63106109/how-to-display-graphs-of-loss-and-accuracy-on-pytorch-using-matplotlib
    '''
    start = time.time()
    criterion = BCELoss() # loss function # TODO: use different loss function
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum) # TODO: use different optimizer
    loss = 0.0
    lossList = []

    for epoch in range(epochs):
        print(f'Epoch { epoch+1 }/{ epochs }')
        model.train()

        bestVal = 0

        epochLoss = []
        # iterate training data loader
        for i, (inputs, targets) in enumerate(trainDL):
            optimizer.zero_grad() # set to zero gradients
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1) # get class labels
            loss = criterion(outputs, targets)
            loss.backward() # set the loss to back propagate through the network updating the weights as it goes
            optimizer.step()

            epochLoss.append(loss.item())
        # torch.save(model, savedPath)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, savedPath)

        # store best model
        if epoch % 10 == 0:
            yhat = model(inputs) # evaluate the model on the test set
            yhat = yhat.detach().numpy()

            actual = targets.numpy() # extract the weights to ndarray instead of tensor
            actual = actual.reshape((len(actual), 1))

            yhat = yhat.round()

            preds = [yhat]
            actuals = [actual]
            preds, actuals = vstack(preds), vstack(actuals)
            cm = numpy.zeros((2, 2)) # define confusion matrix
            for i in range(len(preds)):
                cm[int(actuals[i][0]), int(preds[i][0])] += 1
            accuracy = cm.diagonal().sum() / len(preds)

            if accuracy >= bestVal:
                bestModel = copy.deepcopy(model)
                bestVal = accuracy

                # torch.save(model, savedPath)
                # save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                }, savedPath)

        lossList.append(sum(epochLoss) / len(epochLoss))

    timeDelta = time.time() - start
    print(f'Trainging complete in { timeDelta // 60 }, { timeDelta % 60 }s')

    savePlots(lossList)

    return model

def savePlots(trainLoss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # print(trainLoss)
    plt.figure(figsize=(10, 7))
    plt.plot(
        trainLoss, color='orange', linestyle='-', 
        label='train loss'
    )
    # plt.plot(
    #     valid_loss, color='red', linestyle='-', 
    #     label='validataion loss'
    # )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')


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

    preds, actuals = vstack(preds), vstack(actuals)
    cm = numpy.zeros((2, 2)) # define confusion matrix
    for i in range(len(preds)):
        cm[int(actuals[i][0]), int(preds[i][0])] += 1
    accuracy = cm.diagonal().sum() / len(preds)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion matrix:\n {cm}')
    tn, fp, fn, tp = cm.ravel() # cm.ravel() - Return a contiguous flattened array.
    total = sum(cm.ravel())

    metrics = {
        'accuracy': accuracy,
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
    '''
    return a prediction for each input passed into the model
    '''
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat   

def prepareDataset(path):
    dataset = ThryoidCSVDataset(path)
    train, test = dataset.splitData(ratio=0.3)

    # data loaders
    trainDL = DataLoader(train, batch_size=32, shuffle=True)
    testDL = DataLoader(test, batch_size=1024, shuffle=False)

    return trainDL, testDL

plt.style.use('ggplot')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(10)

trainDL, testDL = prepareDataset('https://raw.githubusercontent.com/StatsGary/Data/main/thyroid_raw.csv') # load data

# train the model
model = ThyroidMLP(26)
trainModel(trainDL=trainDL, model=model, epochs=1000, lr=1e-3)

results = evaluateModel(testDL, model, beta=1)

# write metrics to a csv
modelMetrics = results[0]
metricsDF = pd.DataFrame.from_dict(modelMetrics, orient='index', columns=['metric'])
metricsDF.reset_index(inplace=True)
metricsDF.to_csv('confusion_matrix_thyroid.csv', index=False)

# prediction
# true
row = [0.8408678952719717,0.7480132415430958,-0.3366221139379705,-0.0938130059640389,-0.1101874782051067,-0.2098160394213988,-0.1260114177378201,-0.1118651062104989,-0.1274917875477927,-0.240146053214037,-0.2574472174396955,-0.0715198539852151,-0.0855764265990022,-0.1493202733578882,-0.0190692517849118,-0.2590488060984638,0.0,-0.1753175780014474,0.0,-0.9782211033008232,0.0,-1.3237957945784953,0.0,-0.6384998731458282,0.0,-1.209042232192488]
yhat = predict(row, model)
print(f'Predicted: { yhat.round(3) } (class={ int(yhat.round()) })')

# true
row = [1.2339564002880206,0.7480132415430958,-0.3366221139379705,-0.0938130059640389,-0.1101874782051067,-0.2098160394213988,-0.1260114177378201,-0.1118651062104989,-0.1274917875477927,-0.240146053214037,-0.2574472174396955,-0.0715198539852151,-0.0855764265990022,-0.1493202733578882,-0.0190692517849118,-0.2590488060984638,0.0,-0.1840637959031139,0.0,-0.1257019695838398,0.0,-0.7603760324639954,0.0,-0.8422102564436934,0.0,-0.3546744383145823]
yhat = predict(row, model)
print(f'Predicted: { yhat.round(3) } (class={ int(yhat.round()) })')

# false
row = [-0.7806221879192302,0.7480132415430958,-0.3366221139379705,-0.0938130059640389,-0.1101874782051067,-0.2098160394213988,-0.1260114177378201,-0.1118651062104989,-0.1274917875477927,-0.240146053214037,-0.2574472174396955,-0.0715198539852151,-0.0855764265990022,-0.1493202733578882,-0.0190692517849118,-0.2590488060984638,0.0,-0.1363571528031149,0.0,-0.1257019695838398,0.0,-0.3659821989838455,0.0,0.2781968516945646,0.0,-0.5987795222796983]
yhat = predict(row, model)
print(f'Predicted: { yhat.round(3) } (class={ int(yhat.round()) })')