from funClass import *
import torch
import torch.optim as optim

xRan, yRan = dataLoad(2, 4, 5)

rnnModel = NextWordRNN(xRan.shape)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(rnnModel.parameters(), lr=0.001)
# ====================================================
# ====================================================
# numEpochs = 1
# for epoch in range(numEpochs):
#     rnnModel.train()
#     for i, data
optimizer.zero_grad()

predict, hidden = rnnModel(xRan)
# yRan = torch.randint(0, 3, (1,)).long()
# loss = criterion(predict, torch.tensor([1]).long())
print(yRan)
loss = criterion(predict, yRan)