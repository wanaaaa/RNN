from funClass import *
import torch
import torch.optim as optim

batchSize = 10
xRan, yRan = dataLoad(100, 4, 5)
xyDataLoader = myDataLoaderFun(xRan, yRan, batch_size=batchSize)

rnnModel = NextWordRNN(xRan.shape)
rnnModel.to(device='cuda')
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(rnnModel.parameters(), lr=0.001)

numEpochs = 1
for epoch in range(numEpochs):
    rnnModel.train()
    train_running_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(xyDataLoader):
        optimizer.zero_grad()

        rnnModel.init_hidden(batchSize)

        xTrain, yTrain = data
        yPredict, hidden = rnnModel(xTrain)

        loss = criterion(yPredict, yTrain)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(yPredict, yTrain, batchSize)

    rnnModel.eval()
    print('epoch ->', epoch, ' Loss->', train_running_loss/i, ' train accuracy->', train_acc/i)

    torch.cuda.empty_cache()

rnnModel.eval()

xTest, yTest = dataLoad(1, 4, 5)
xTest = xTest
yPredict , hidden = rnnModel(xTest)
print(yPredict)

# numEpochs = 1
# for epoch in range(numEpochs):
#     # rnnModel.train()
#     for i, xData in enumerate(xRan):
#         xData = torch.unsqueeze(xData, 0)
#         predict, hidden = rnnModel(xData)
#         loss = criterion(predict, yRan[i:i+1])
