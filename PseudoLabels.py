import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

classifiers = {"KNN": KNeighborsClassifier(), "MLP": MLPClassifier(), "SVC": SVC(), 
               "Desicion Tree":  DecisionTreeClassifier(), "Random Forest Classifier": RandomForestClassifier(),
               "Gaussian":  GaussianNB(), "QDA": QuadraticDiscriminantAnalysis(), 
               "Logistic Regression": LogisticRegression()}

class PseudoLabels:
    def __init__(self, X, y, num_labeled):
        self.X_labeled = X[:num_labeled]
        self.y_labeled = y[:num_labeled]

        self.X_unlabeled = X[num_labeled:]   
        
    
    def label_propagation(self):
        model = LabelPropagation(kernel='knn')

        model.fit(self.X_labeled, self.y_labeled)
        #print("Accuracy (without pseudo-labels): ",  model.score(X_test, y_test))
        
        pseudo_labels = model.predict(self.X_unlabeled)
            
        X_combined = np.concatenate((self.X_labeled, self.X_unlabeled))
        y_combined = np.concatenate((self.y_labeled, pseudo_labels))

        model.fit(X_combined, y_combined)

        #accuracy = model.score(X_test, y_test)
        #print('Accuracy:', model.score(X_test, y_test))
    
    def label_spreading(self):
        pass
    
    def self_training(self, model_name, model):
        models_statistics = []
        
        kfold = KFold(n_splits=10)
        
        scores = []
        for train_index, val_index in kfold.split(self.X_labeled):
            X_train_fold, X_val = self.X_labeled[train_index], self.X_labeled[val_index]
            y_train_fold, y_val = self.y_labeled[train_index], self.y_labeled[val_index]
            
            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        models_statistics.append([model_name, np.mean(scores)])
        
        pseudo_labels = model.predict(self.X_unlabeled)
            
        # Combine the labeled and pseudo labeled data
        X_combined = np.concatenate((self.X_labeled, self.X_unlabeled))
        y_combined = np.concatenate((self.y_labeled, pseudo_labels))

        model.fit(X_combined, y_combined)
        
        scores = []
        for train_index, val_index in kfold.split(X_combined):
            X_train_fold, X_val = X_combined[train_index], X_combined[val_index]
            y_train_fold, y_val = y_combined[train_index], y_combined[val_index]
            
            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        models_statistics.append([model_name + "(with pseudo-labels)", np.mean(scores)])
        
        return models_statistics
      
    
data = pd.read_csv("data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X = np.array(X)
y = np.array(y)

pseudo_labels = PseudoLabels(X, y, 100)
pseudo_labels.label_propagation()

models_statistics = []

for clf_name in classifiers:
    for model_stat in pseudo_labels.self_training(clf_name, classifiers[clf_name]):
        models_statistics.append(model_stat)
        
print(models_statistics)