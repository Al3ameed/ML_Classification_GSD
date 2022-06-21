import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
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

# 3 - EDA (Exploratory Data Analysis) Process and Feature Engineering
for column in numericalFeatures:
    plt.figure(column, figsize=(5, 5))
    plt.title(column)
    dataSet[column].plot(kind='hist')
    plt.tight_layout()
plt.show()
for column in categoricalFeatures:
    plt.figure(column, figsize=(5, 5))
    plt.title(column)
    dataSet[column].value_counts().plot(kind='pie')
    plt.tight_layout()
plt.show()

plt.figure(figsize=(13, 9))
correlation = dataSet.corr().round(1)
sns.heatmap(correlation, cmap='GnBu', annot=True)
plt.tight_layout()
plt.show()

# 4 - Split Dataset into Training and Testing Set
Cols = dataSet.shape[1]
XCols = dataSet.shape[1]
X = dataSet.iloc[:, 0: Cols - 1]
Y = dataSet.iloc[:, Cols - 1: Cols]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, train_size=0.80, random_state=42)

# 5 - [Building Model] Machine Learning Classification PipeLine
model_pipeline = [
    LogisticRegression(solver='liblinear'),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    GaussianNB()
]
model_list = ['Logistic Regression', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes']
acc_list = []
auc_list = []
cm_list = []
recall_list = []
precision_list = []
for model in model_pipeline:
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    # Evaluation
    cm_list.append(confusion_matrix(Y_test, y_pred))
    acc_list.append(metrics.accuracy_score(Y_test, y_pred))
    fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred)
    auc_list.append(round(metrics.auc(fpr, tpr), 2))
    recall_list.append(round(metrics.recall_score(Y_test, y_pred), 2))
    precision_list.append(round(metrics.precision_score(Y_test, y_pred), 2))
fig = plt.figure(figsize=(18, 10))
for i in range(len(cm_list)):
    cm = cm_list[i]
    model = model_list[i]
    sub = fig.add_subplot(2, 3, i + 1).set_title(model)
    cm_plot = sns.heatmap(cm, cmap='GnBu', annot=True)
    cm_plot.set_ylabel('Predicted Value ')
    cm_plot.set_xlabel('Actual Value')
plt.tight_layout()
plt.show()
results = pd.DataFrame({
    'Model': model_list,
    'Accuracy': acc_list,
    'AUC': auc_list,
    'Recall': recall_list,
    'Precision': precision_list
})
print(results.to_string(max_cols=dataSet.shape[1]))
