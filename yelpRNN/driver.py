from funClass import *
import torch
import torch.optim as optim

# createWordVecModelFun('./wordDataJson/review10th.json')

# ================================
wordSentenceDbLi = dataToXYListRead('./wordDataJson/review1000th.json')
trainXYLi, testXYLi = splitToTrainTestFun(wordSentenceDbLi)
# print(testXYLi)
xVec, yVec = processData(trainXYLi)

batchSize = 7
xyTainSet = myDataLoader(xVec, yVec, batchSize=batchSize)
rnnModel = NextWordRNN(batchSize=batchSize, numSequence=4, numFeature=10)
rnnModel.to(device='cuda')
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(rnnModel.parameters(), lr=0.001)

numEpochs = 1
for epoch in range(numEpochs):
    rnnModel.train()
    train_running_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(xyTainSet):
        optimizer.zero_grad()

        rnnModel.init_hidden(batchSize=batchSize)

        xTrain, yTrain = data
        yPredict, hidden = rnnModel(xTrain)

        loss = criterion(yPredict, yTrain)
        loss.backward()
        optimizer.step()
    #
        train_running_loss += loss.detach().item()
    #     train_acc += get_accuracy(yPredict, yTrain, batchSize)
    #
    # print('epoch ->', epoch, ' Loss->', train_running_loss/i, ' train accuracy->', train_acc/i)






