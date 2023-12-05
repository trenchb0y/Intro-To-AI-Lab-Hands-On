import pandas as pd
import math

def read_excel(path, sheet_target): # read excel data
    data = pd.read_excel(path, sheet_name=sheet_target)
    return pd.DataFrame(data)

dataTrain = read_excel("newTrain.xlsx",sheet_target="train")
dataTest = read_excel("newTest.xlsx",sheet_target="Sheet1")
dataTruth = read_excel("newTestGroundTruth.xlsx",sheet_target="Sheet1")

# dataTrain.info()
# dataTest.info()
# dataTruth.info()

dataTrain = dataTrain.drop(["Unnamed: 0"],axis=1)
dataTest = dataTest.drop(["Unnamed: 0"],axis=1)
dataTruth = dataTruth.drop(["Unnamed: 0"],axis=1)

# print(dataTest.head(5))
# print(dataTruth.head(5))

def normalizationMinMax(allData, columnTarget):
    for column in columnTarget:
        allData[column] = (allData[column] - min(allData[column])) / (max(allData[column]) - min(allData[column]))
    return  allData

testStartID = dataTrain.index.stop 
# print(testStartID)# Since we will combine them to do normalization
allData = pd.concat([dataTrain, dataTest]) # Combine them
allData = normalizationMinMax(allData, columnTarget=["x1","x2","x3"])
normTrain, normTest = allData.iloc[:testStartID].drop('id', axis=1), allData.iloc[testStartID:].drop('id', axis=1)

# print(normTrain)
# print(normTest)

def standardization(allData, columnTarget):
    for column in columnTarget:
        allData[column] = (allData[column] - allData[column].mean()) / allData[column].std()
   
    return allData

testStartID = dataTrain.index.stop # Since we will combine them to do normalization
allData = pd.concat([dataTrain, dataTest]) # Combine them
allData = standardization(allData, columnTarget=["x1","x2","x3"])
stdTrain, stdTest = allData.iloc[:testStartID].drop('id', axis=1), allData.iloc[testStartID:].drop('id', axis=1)

# print(stdTrain)
# print(stdTest)

def splitTruth(dataTrain, columnTarget):
    truthData = []
    for truth in dataTrain[columnTarget].unique():
        truthData.append(dataTrain.where(dataTrain[columnTarget] == truth).dropna())
    return truthData

yesData, noData = splitTruth(dataTrain, columnTarget='y')

# print(type(dataTrain))
# print(yesData)
# print(noData)

def find_mean(yesData, noData, columnTarget):
    yesMean = dict()
    noMean = dict()
    for column in columnTarget:
        yesMean[column] = yesData[column].mean()  # it is a dataframe variable
        noMean[column] = noData[column].mean() # it is a dataframe variable
    return yesMean, noMean

yesMean, noMean = find_mean(yesData, noData, columnTarget=["x1","x2","x3"])

print(f"Mean Result\n1 : {yesMean}\n0 : {noMean}")

def find_std(yesData, noData, columnTarget):
    yesStd = dict()
    noStd = dict()
    for column in columnTarget:
        yesStd[column] = yesData[column].std() # it is a dataframe variable
        noStd[column] = noData[column].std() # it is a dataframe variable

    return yesStd, noStd

yesStd, noStd = find_std(yesData, noData, columnTarget=["x1","x2","x3"])

print(f"Standard Deviation Result\n1 : {yesStd}\n0 : {noStd}")

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
        print("\nCannot process the confussion matrix with unknown Ground Truth!")

def doPrediction(yesMean, yesStd, noMean, noStd, target, columnTarget, truthColumn):
    result = []
    for i in range(len(target)):
        yesResult = 1
        noResult = 1
        for column in columnTarget:
            yesResult *= calc_probability(yesMean[column], yesStd[column], target[column].iloc[i])
            noResult *= calc_probability(noMean[column], noStd[column], target[column].iloc[i])
        result.append({'ID' : target['id'].iloc[i], 'Yes Probability' : "{}".format(yesResult), 'No Probability' : "{}".format(noResult),
                        'Prediction Result' : int(yesResult > noResult), 
                       'Ground Truth' : target[truthColumn].iloc[i]})

    return result

result = []
target = dataTest # The data you want to predict its truth value ('y' column)

result = doPrediction(yesMean, yesStd, noMean, noStd, target, columnTarget=["x1","x2","x3"], truthColumn='y')

for p in result:
    print(p)

confussionMatrix(result)

def folding(dataset, trainingPercentage, location, shuffle:bool):
    lengthTraining = int(len(dataset)*trainingPercentage/100)
    # randomize the the data position
    if(shuffle):
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    train = []
    validation = []
    if(location == 'left'):
        train, validation = dataset.iloc[:lengthTraining].reset_index(drop=True), dataset.iloc[lengthTraining:].reset_index(drop=True)
    elif(location == 'right'):
        validation,train = dataset.iloc[:abs(lengthTraining-len(dataset))].reset_index(drop=True), dataset.iloc[abs(lengthTraining-len(dataset)):].reset_index(drop=True)
    elif(location == 'middle'):
        train = dataset.iloc[int(abs(lengthTraining-len(dataset))/2):len(dataset)-int(abs(lengthTraining-len(dataset))/2)]
        validation = pd.concat([dataset.iloc[:int(abs(lengthTraining-len(dataset))/2)],dataset.iloc[len(dataset)-int(abs(lengthTraining-len(dataset))/2):]])
    return train, validation

trainData, validationData = folding(dataTrain.copy(), trainingPercentage=78, location="middle", shuffle=True)

print(trainData)
print(validationData)

yesDataTrain, noDataTrain = splitTruth(trainData, columnTarget='y')

print(yesDataTrain)
print(noDataTrain)

# Find mean
yesMeanTrain, noMeanTrain = find_mean(yesDataTrain, noDataTrain, columnTarget=["x1","x2","x3"])
# Find standard deviation
yesStdTrain, noStdTrain = find_std(yesDataTrain, noDataTrain, columnTarget=["x1","x2","x3"])

result = doPrediction(yesMeanTrain, yesStdTrain, noMeanTrain, noStdTrain, target, columnTarget=["x1","x2","x3"], truthColumn='y')
target = validationData # your ground truth data

for p in result:
    print(p)

confussionMatrix(result)

result = []
target = dataTest # your ground truth data

result = doPrediction(yesMeanTrain, yesStdTrain, noMeanTrain, noStdTrain, target, columnTarget=["x1","x2","x3"], truthColumn="y")

for p in result:
    print(p)

confussionMatrix(result)