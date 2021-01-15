import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

def dataLoad(batch, sequence, feature):
    xRan = torch.rand(batch, sequence, feature, device='cuda')  # batch, sequence, feature
    # yRan = torch.rand(batch, feature)
    yRan = torch.randint(0, feature, (batch,), device='cuda').long()

    return xRan, yRan

class NextWordRNN(torch.nn.Module):
    def __init__(self, xShape):
        super(NextWordRNN, self).__init__()

        self.hiddenDim = 3
        self.numLayers = 1
        # self.inputSize = 5 # number of input in a x
        self.inputSize = xShape[2] # number of input in a x
        self.NumSequence = xShape[1]
        self.NumFeature = xShape[2]

        self.rnn = torch.nn.RNN(self.inputSize, self.hiddenDim, batch_first=True)
        self.fc = torch.nn.Linear(self.hiddenDim, 5)
        self.fcHidden = torch.nn.Linear(self.hiddenDim, 1)
        self.fcFinal = torch.nn.Linear(self.NumSequence, self.NumFeature)

    def forward(self, x):
        batchSize = len(x)
        hidden = self.init_hidden(batchSize)

        out, hidden = self.rnn(x, hidden)
        out = self.fcHidden(out)
        out = out.squeeze()
        out = self.fcFinal(out)

        # print("before view==>", out.shape )
        out = out.view(-1, self.NumFeature)
        # print("after view==>", out.shape )

        return out, hidden

    def init_hidden(self, batchSize):
        hidden = torch.zeros(self.numLayers, batchSize, self.hiddenDim, device='cuda')
        return hidden

def myDataLoaderFun(xTensor, yTensor, batch_size):
    xyTensor = TensorDataset(xTensor, yTensor)
    trainSampler = RandomSampler(xyTensor)
    xyDataLoader = DataLoader(xyTensor, sampler=trainSampler, batch_size=batch_size)

    return xyDataLoader

def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()