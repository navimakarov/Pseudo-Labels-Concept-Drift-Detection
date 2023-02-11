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

from sklearn.model_selection import KFold


models = {"KNN": KNeighborsClassifier(), "MLP": MLPClassifier(), "SVC": SVC(probability=True), 
               "Desicion Tree":  DecisionTreeClassifier(), "Random Forest Classifier": RandomForestClassifier(),
               "Gaussian":  GaussianNB(), "QDA": QuadraticDiscriminantAnalysis(), 
               "Logistic Regression": LogisticRegression(), "Label Propagation": LabelPropagation(kernel='knn'),
               "Label Spreading": LabelSpreading(kernel='knn')}

class PseudoLabels:
    def __init__(self, X, y, num_labeled, threshold=0.0):
        self.X_labeled = X[:num_labeled]
        self.y_labeled = y[:num_labeled]

        self.X_unlabeled = X[num_labeled:]  
        
        self.k_fold = KFold(n_splits=10)
        self.threshold = threshold
        
    
    def k_fold_cross_validation(self, model, X, Y):
        scores = []
        for train_index, test_index in self.k_fold.split(X):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            
            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_test_fold, y_test_fold)
            scores.append(score)
            
        accuracy = np.mean(scores)
        return accuracy
        
    
    def train_supervised_model(self, model_name):
        model = models[model_name]
        
        accuracy = self.k_fold_cross_validation(model, self.X_labeled, self.y_labeled)
        
        return {"model_name": model_name, "model": model, 
                "accuracy": accuracy, "pseudo_labels": False}
    
    
    def train_semi_supervised_model(self, model_name, supervised_model):
        model = models[model_name]
        
        probabilities = supervised_model.predict_proba(self.X_unlabeled)
        pseudo_labels = np.argmax(probabilities, axis=1)
    
        high_probability_indices = np.where(np.max(probabilities, axis=1) >= self.threshold)[0]
        pseudo_labels = pseudo_labels[high_probability_indices]
        X_unlabeled = self.X_unlabeled[high_probability_indices]
    
        # Combine the labeled and pseudo labeled data
        X_combined = np.concatenate((self.X_labeled, X_unlabeled))
        y_combined = np.concatenate((self.y_labeled, pseudo_labels))
    
        accuracy = self.k_fold_cross_validation(model, X_combined, y_combined)
    
        return {"model_name": model_name, "model": model, 
                "accuracy": accuracy, "pseudo_labels": True}
      
    
data = pd.read_csv("data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X = np.array(X)
y = np.array(y)

pseudo_labels = PseudoLabels(X, y, 10, 0.5)

models_statistics = []

for model_name in models:
    supervised_model_overview = pseudo_labels.train_supervised_model(model_name)
    models_statistics.append(supervised_model_overview)
    
    semi_supervised_model_overview = pseudo_labels.train_semi_supervised_model(model_name, supervised_model_overview["model"])
    models_statistics.append(semi_supervised_model_overview)
        
print(models_statistics)