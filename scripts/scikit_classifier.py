import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# SVM training
def train_svm(train_data, train_label, splitter, tuning_params):
    svc = svm.SVC()
    
    clf = GridSearchCV(estimator=svc, param_grid=tuning_params, cv=splitter)
    clf.fit(train_data, train_label)
    
    # Return best model
    return clf.best_estimator_

# Random Forest training
def train_rf(train_data, train_label, splitter, tuning_params, seed=0):
    rf = RandomForestClassifier(random_state=seed)

    clf = GridSearchCV(estimator=rf, param_grid=tuning_params, cv=splitter)
    clf.fit(train_data, train_label)
    
    # Return best model
    return clf.best_estimator_

# Calculate confusion matrix from trained model
def prediction_matrix(test_data, test_label, classifier, num_classes):
    pred = classifier.predict(test_data)
    
    matrix = np.zeros((num_classes, num_classes))

    for i in range(len(test_label)):
        row = test_label[i]
        col = pred[i]
        matrix[row][col] += 1
    return matrix
