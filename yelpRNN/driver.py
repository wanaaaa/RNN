from funClass import *

# createWordVecModelFun('./wordDataJson/review10th.json')

# ================================
# ================================
wordSentenceDbLi = dataToXYListRead('./wordDataJson/review1000th.json')
trainXYLi, testXYLi = splitToTrainTestFun(wordSentenceDbLi)
# print(testXYLi)
xVec, yVec = processData(trainXYLi)
