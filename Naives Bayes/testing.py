import pandas as pd
import math

exerciseData = pd.read_csv("Naive-Bayes-Classification-Data.csv")

def normalizationMinMax(allData, columnTarget):
    for column in columnTarget:
        allData[column] = (allData[column] - min(allData[column])) / (max(allData[column]) - min(allData[column]))
    return allData

testStartID = exerciseData.index.stop  # Since we will combine them to do normalization
normalizedData = normalizationMinMax(exerciseData, columnTarget=["glucose", "bloodpressure"])

normTrain, normTest = normalizedData.iloc[:testStartID].drop('id', axis=1), normalizedData.iloc[testStartID:].drop('id', axis=1)

def splitTruth(normTrain, columnTarget):
    truthData = []
    for truth in normTrain[columnTarget].unique():
        truthData.append(normTrain.where(normTrain[columnTarget] == truth).dropna())
    return truthData

noData, yesData = splitTruth(normTrain, columnTarget='diabetes')

def find_mean(yesData, noData, columnTarget):
    yesMean = dict()
    noMean = dict()
    for column in columnTarget:
        yesMean[column] = yesData[column].mean()
        noMean[column] = noData[column].mean()
    return yesMean, noMean

yesMean, noMean = find_mean(yesData, noData, columnTarget=["glucose","bloodpressure"])

def find_std(yesData, noData, columnTarget):
    yesStd = dict()
    noStd = dict()
    for column in columnTarget:
        yesStd[column] = yesData[column].std()
        noStd[column] = noData[column].std()
    return yesStd, noStd

yesStd, noStd = find_std(yesData, noData, columnTarget=["glucose","bloodpressure"])

def calc_probability(mean, std, x):
    exponent = math.exp(-((x-mean)**2 / (2 * std**2)))
    return (1/(math.sqrt(2*math.pi)*std)) * exponent

def confussionMatrix(result):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    x = True

    for i in result:
        if(i['Ground Truth'] == '?'):
            x = False
            break
        elif((i['Prediction Result'] == 1)and(i['Prediction Result'] == i['Ground Truth'])):
            TP += 1
        elif((i['Prediction Result'] == 0)and(i['Prediction Result'] == i['Ground Truth'])):
            TN += 1
        elif((i['Prediction Result'] == 1)and(i['Prediction Result'] != i['Ground Truth'])):
            FP += 1
        elif((i['Prediction Result'] == 0)and(i['Prediction Result'] != i['Ground Truth'])):
            FN += 1

    if(x):
        print(f"\nTP : {TP} FN : {FN}\nTN : {FN} FN : {FN}")
        print(f"Accuracy : {((TP+TN)/(TP+TN+FN+FP))*100}%")
        print(f"Precission : {((TP)/(TP+FP))*100}%")
        print(f"Recall : {((TP)/(TP+FN))*100}%")
    else:
        print("\nCannot process the confusion matrix with unknown Ground Truth!")

def doPrediction(yesMean, yesStd, noMean, noStd, target, columnTarget, truthColumn):
    result = []
    for i in range(len(target)):
        yesResult = 1
        noResult = 1 
        for column in columnTarget:
            yesResult *= calc_probability(yesMean[column], yesStd[column], target[column].iloc[i])
            noResult *= calc_probability(noMean[column], noStd[column], target[column].iloc[i])
        result.append({'Yes Probability': "{}".format(yesResult), 'No Probability': "{}".format(noResult),
                       'Prediction Result': int(yesResult > noResult), 
                       'Ground Truth': target[truthColumn].iloc[i]})
    return result

result = doPrediction(yesMean, yesStd, noMean, noStd, target=normTrain, columnTarget=["glucose","bloodpressure"], truthColumn="diabetes")

for p in result:
    print(p)

confussionMatrix(result)
