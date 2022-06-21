import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# 1. Importing Dataset
dataSet = pd.read_csv("dataSet.csv", encoding='latin-1')
print(dataSet.describe().to_string(max_cols=dataSet.shape[1]))

# 2. Data PreProcessing [ Cleaning, Missing Values, Feature Scaling]
numericalFeatures = []
categoricalFeatures = []
missingCount = dataSet.isnull().sum()  # count of missing Values
valueCount = dataSet.isnull().count()  # count of all values
missingPercentage = round(missingCount / valueCount * 100, 1)  # percentage of missing values in dataSet
missingDf = pd.DataFrame({
    'count': missingCount,
    'percentage': missingPercentage
})  # create DataFrame to handle missing Values
# drop columns with large number of missing values + will not affect our Model
dataSet = dataSet.drop(['Name', 'Timestamp', 'Job'], axis=1)
for column in dataSet:
    if is_numeric_dtype(dataSet[column]):
        numericalFeatures.append(column)
    elif is_string_dtype(dataSet[column]):
        categoricalFeatures.append(column)
# Numerical Feature: replace missing values with the mean
dataSet.fillna(0, inplace=True)
# Categorical Variable
for i in categoricalFeatures:
    if dataSet[i].isnull().any():
        dataSet[i].fillna("unKnown", inplace=True)

# 3 - Feature Transformation & Categorical Feature Encoding
for column in categoricalFeatures:
    dataSet[column] = LabelEncoder().fit_transform(dataSet[column])


Cols = dataSet.shape[1]
XCols = dataSet.shape[1]
X = np.array(dataSet.iloc[:, 0: Cols - 1])
Y = np.array(dataSet.iloc[:, Cols - 1: Cols])
results = pd.DataFrame()

lg_accuracy = []
lg_auc = []
lg_recall = []
lg_precision = []

svc_accuracy = []
svc_auc = []
svc_recall = []
svc_precision = []

knn_accuracy = []
knn_auc = []
knn_recall = []
knn_precision = []

nb_accuracy = []
nb_auc = []
nb_recall = []
nb_precision = []

rf_accuracy = []
rf_auc = []
rf_recall = []
rf_precision = []

dt_accuracy = []
dt_auc = []
dt_recall = []
dt_precision = []

model_pipeline = [
    LogisticRegression(solver='liblinear'),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB()
]
model_list = ['Logistic Regression', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes']
for model in model_pipeline:
    accuracy = np.mean(cross_val_score(model, X, Y, scoring='accuracy'))
    auc = np.mean(cross_val_score(model, X, Y, scoring='roc_auc'))
    recall = np.mean(cross_val_score(model, X, Y, scoring='recall'))
    precision = np.mean(cross_val_score(model, X, Y, scoring='precision'))
    results = results.append({
        "Model": model,
        "Accuracy": accuracy,
        "AUC": auc,
        "Recall": recall,
        "Precision": precision
    }, ignore_index=True)
print(results.to_string(max_cols=dataSet.shape[1]))
