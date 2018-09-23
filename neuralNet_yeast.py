import numpy as np
import pandas as pd 
import sys 
import matplotlib.pyplot as plt 
from pandas.plotting import scatter_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Loading of dataset 
header = ['SEQ_NAME','MCG','GVH', 'ALM','MIT','ERL','POX','VAC','NUC', 'CLASS']
yeast = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', header = None, sep = '\s+', names = ['SEQ_NAME','MCG','GVH', 'ALM','MIT','ERL','POX','VAC','NUC', 'CLASS'], usecols = ['MCG','GVH', 'ALM','MIT','ERL','POX','VAC','NUC','CLASS'])
yeast.shape
df = pd.DataFrame(yeast)

# Preprocessing
df["CLASS"] = df["CLASS"].astype('category')
df["CLASS"] = df["CLASS"].cat.codes

# Histogram of all the features
df.iloc[:, 0:8].hist()

# Plotting the correlations
correlations = df.iloc[:, 0:8].corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.iloc[:, 0:8].columns)
ax.set_yticklabels(df.iloc[:, 0:8].columns)
# plt.show()

scatter_matrix(df.iloc[:, 0:8])
# plt.show()

#Training and Test Data
X = df.iloc[:, 0:8].values
y = df.iloc[:, 8:9].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X,y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Building neuralnet with MLPClassifier
mlp = MLPClassifier( warm_start = True, hidden_layer_sizes = (24, 24, 24), activation = 'tanh', solver = 'sgd', momentum = 0.95, learning_rate = 'adaptive',  max_iter = 3000, alpha = 0.00025)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

#Result Display 
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))
print("CLASSIFICATION REPORT")
print(classification_report(y_test, predictions))
print("ACCURACY SCORE")
print(accuracy_score(y_test, predictions))
print("TRAINING SCORE")
print(mlp.score(X_train,y_train))
