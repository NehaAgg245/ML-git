import pandas as pd 
import sys 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, shuffle = True)
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
	tuned_parameters = [{'hidden_layer_sizes':[(20,25),(10,20,30),(2,8)],
	'activation':['identity','logistic','tanh','relu'],
	'learning_rate':['constant', 'invscaling', 'adaptive'],
	'learning_rate_init':[0.001,0.005,0.009],
	'alpha':[0.0001,0.001,0.0005]}]
	mlp = MLPClassifier(warm_start = True,  max_iter = 5000)
	clf = GridSearchCV(mlp, tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#Support Vector Machine GridSearch
def svm(score):
	return clf

#Gaussian Naive Bayes GridSearch0.6,0.2,0.05,0.001,0.001,0.023,0.05,0.075
def naive_bayes(score):
	tuned_parameters = [{'priors':[0.6,0.2,0.05,0.001,0.001,0.023,0.05,0.075]}]
	clf = GridSearchCV(GaussianNB(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#Logistic Regression GridSearch
def logistic_regression(score):
	tuned_parameters = [{
	'tol' : [1e-4,1e-3,1e-5],
	'C': [1,0.5,1.5],
	'solver':['newton-cg', 'lbfgs', 'saga', 'sag'],
	'max_iter' : [5000],
	'multi_class':['ovr', 'multinomial','auto']}]
	clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#KNN GridSearch
def knn(score):
	tuned_parameters = [{'n_neighbors':[5,15,25,35],
	'weights':['uniform', 'distance'],
	'algorithm':['auto','ball_tree','kd_tree','brute'],
	'p':[1,2,3,5]}]
	clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#Bagging GridSearch
def bagging(score):
	tuned_parameters = [{'n_estimators': [10,15,5,25],
	'max_samples':[0.5,0.2,1.0],
	'max_features':[0.2,0.5,1],
	'random_state':[0, None]}]
	clf = GridSearchCV(BaggingClassifier(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#Random Forest GridSearch 
def random_forest(score):
	tuned_parameters = [{'n_estimators': [5,10,25,50],
	'criterion':['gini', 'entropy'],
	'max_depth' : [1,3,5,7,10],
	'max_features' : [4, 'sqrt', 'log2', None]}]
	clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#Adaboost GridSearch 
def adaboost(score):
	tuned_parameters = [{'n_estimators':[50, 100, 25, 200],
	'learning_rate' : [1, 0.5, 1.5],
	'algorithm' :['SAMME', 'SAMME.R'],
	'random_state':[0, None]}]
	clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#Gradient Boosting GridSearch 
def gradient_boosting(score):
	tuned_parameters = [{'loss':['deviance'],
	'learning_rate':[0.1,0.5,0.01,0.9,0.05],
	'max_depth':[3,1,7,9],
	'min_samples_split' :[2,5,10],
	'max_features': [7, 'sqrt', 'log2']}]
	clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#XGBoost GridSearch
def xgboost(score):
	tuned_parameters = [{'learning_rate':[0.1,0.5,0.01,0.9,0.05],
	'n_estimators':[100,50,1000],
	'booster':['gbtree','gblinear','dart'],
	'max_delta_step':[0,1,5]}]
	clf = GridSearchCV(XGBClassifier(), tuned_parameters, cv = 5, scoring = '%s' % score)
	return clf

#Calling the model
scores = ['accuracy']
model = input("Enter the model name (Consult readme file): ")

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = xgboost(score)
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