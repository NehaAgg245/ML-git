import pandas as pd 
import sys 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#Loading of dataset 
header = ['Seq_Name','MCG','GVH','LIP','CHG','AAC','ALM1','ALM2', 'CLASS']
yeast = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data', header = None, names = header, sep = '\s+', usecols = ['MCG','GVH','LIP','CHG','AAC','ALM1','ALM2', 'CLASS'])
yeast.shape
df = pd.DataFrame(yeast)

# Preprocessing
df["CLASS"] = df["CLASS"].astype('category')
df["CLASS"] = df["CLASS"].cat.codes

#Training and Test Data
X = df.iloc[:, 0:7].values
y = df.iloc[:, 7:8].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X,y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Decision Tree GridSearch
def decision_tree(score):
	tuned_parameters = [{'max_depth' : [1,3,5,7,10], 
	'max_features': [7, 'sqrt', 'log2'], 
	'max_leaf_nodes' :[ None, 300, 250], 
	'min_impurity_decrease' : [0.01, 0.02, 0.05]}]
	clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#Neural Net GridSearch
def neural_net(score):
	return clf

#Support Vector Machine GridSearch
def svm(score):
	return clf

#Gaussian Naive Bayes GridSearch
def naive_bayes(score):
	return clf

#Logistic Regression GridSearch
def logistic_regression(score):
	return clf

#KNN GridSearch
def knn(score):
	return clf

#Bagging GridSearch
def bagging(score):
	return clf

#Random Forest GridSearch 
def random_forest(score):
	return clf

#Adaboost GridSearch 
def adaboost(score):
	return clf

#Gradient Boosting GridSearch 
def gradient_boosting(score):
	return clf

#XGBoost GridSearch
def xgboost(score):
	return clf

#Calling the model
scores = ['accuracy']
model = input("Enter the model name (Consult readme file): ")

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = decision_tree(score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set for " + model + " : ")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set for " + model + " : ")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report for " + model + " : ")
    print()
    print("The model is trained on the full development set for " + model + ".")
    print("The scores are computed on the full evaluation set for " + model + ".")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_pred))