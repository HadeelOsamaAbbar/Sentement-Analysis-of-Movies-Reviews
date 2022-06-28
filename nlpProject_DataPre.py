from base64 import encode
from operator import mod
import re, os
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics, svm, linear_model, naive_bayes

from sklearn.feature_extraction.text import TfidfVectorizer


def readTextFile(file_path):
    f = open(file_path, 'r')
    return f.read()

def deconstructWords(data):
    data = re.sub(r"\'ve", "have", data)
    data = re.sub(r"can\'t", "can not", data)
    data = re.sub(r"n\'t", " not", data)
    data = re.sub(r"\'m", " am", data)
    data = re.sub(r"\'t", " not", data)
    data = re.sub(r"\'ll", " will", data)
    data = re.sub(r"won\'t", " will not", data)
    data = re.sub(r"\'t", " not", data)
    data = re.sub(r"\'re", " are", data)
    return data

def stemmer(data):
    stemm = ""
    Ps = PorterStemmer()
    for word in data.split():
        stemmW=Ps.stem(word)
        stemm+=stemmW+" "
    return stemm

def lemmatizer(data):
    lemma = ""
    wordNet = WordNetLemmatizer()
    for word in data.split():
        lemmaW = wordNet.lemmatize(word)
        lemma+=lemmaW+" "
    return lemma
def createFolder(direct):
    try:
        if not os.path.exists(direct):
            os.makedirs(direct)
    except OSError:
        print("error")
def saveFile(path,fileName, contentt):
    os.makedirs(path,exist_ok=True)
    with open(os.path.join(path,fileName),"w") as f:
        f.write(contentt)
#########################################



eng_stopwords = set(stopwords.words('english'))
## create folder and save files

#createFolder("D:\STUDY\\NLP\\NLP_Project\\txt_sentoken\\data1")
#path = r"D:\STUDY\\NLP\\NLP_Project\\txt_sentoken\\pos"
# os.chdir(path)

# for file in os.listdir():
#     data = readTextFile(file)
#     saveFile("D:\STUDY\\NLP\\NLP_Project\\txt_sentoken\\data1", "pos-" + file, data)

# path = r"D:\STUDY\\NLP\\NLP_Project\\txt_sentoken\\neg"
# os.chdir(path)

# for file in os.listdir():
#     data = readTextFile(file)
#     saveFile("D:\STUDY\\NLP\\NLP_Project\\txt_sentoken\\data1", "neg-"+file, data)



content = []
allReviews = []
labels = []

path = r"D:\STUDY\NLP\NLP_Project\txt_sentoken\data1"
files=os.listdir(r"D:\STUDY\NLP\NLP_Project\txt_sentoken\data1")
random.shuffle(files)
os.chdir(path)
c = 0
for file in files:

    ##read files
    c = c + 1
    data = readTextFile(file)
    # lowerCase
    allReviews.append(data)
    label = file.split('-')
    if (label[0] == "pos"):
        labels.append(1)
    elif (label[0] == "neg"):
        labels.append(0)
    if c == 2000:
        break
dataFrame = pd.DataFrame()
dataFrame["reviews"] = allReviews
dataFrame["label"] = labels
print(dataFrame.head)
# dataFrame = dataFrame.sample(frac=1)  # random the data

allFilteredData = []
labelWords = []
allReviews
for index in range(len(allReviews)):
    data = allReviews[index]
    data = data.lower()

    # remove words with numbers
    data = re.sub("\S*\d\S*", "", data).strip()

    # deconstruct the words
    data = deconstructWords(data)
    ##remove special charachter
    data = re.sub('[^A-Za-z0-9]+', ' ', data)

    str = ""
    #for ele in data:
    str += data+""
    text=""
    # remove stop words
    splits=str.split()
    for word in splits:
        
        if (word.casefold() != " " and word.casefold() != "\n" and word.casefold() not in eng_stopwords):
           
            text+=word +" "
            allFilteredData.append(word)
            labelWords.append(labels[index])

    content.append(text)
    dataTokens = word_tokenize(data)


lemmaWords = []
allLemma=[]

for index in range(len(allReviews)):
   text = lemmatizer(content[index])
   allLemma.append(text)

dataFrame["reviews"] = allLemma
dataFrame["label"] = labels

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(dataFrame["reviews"], dataFrame["label"])

encoder=LabelEncoder()
yTrain=encoder.fit_transform(yTrain)
yTest=encoder.fit_transform(yTest)
tfidf = TfidfVectorizer(analyzer="word", token_pattern=r'\w{1,}', max_features=5500)
tfidf.fit(dataFrame["reviews"])
xTraintfidf = tfidf.transform(xTrain)
xTestfidf = tfidf.transform(xTest)

#models:
def model(classifer,feature_vector_train,labelL,feature_vector_valid,is_neural_net=False):
    classifer.fit(feature_vector_train,labelL)
    predictions=classifer.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions,yTest)
accuracyLinear_model = model(linear_model.LogisticRegression(), xTraintfidf, yTrain, xTestfidf)
print(accuracyLinear_model)

accuracySVM = model(svm.LinearSVC(C=1.5), xTraintfidf, yTrain, xTestfidf)
print(accuracySVM)

accuracyNaiave = model(naive_bayes.MultinomialNB(alpha=0.2), xTraintfidf, yTrain, xTestfidf)
print(accuracyNaiave)
