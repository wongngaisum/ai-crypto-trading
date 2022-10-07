
import pandas as pd
from sklearn.model_selection import train_test_split
from model_matrix import get_clean_Xy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

def get_metrics(data, classifier, test_size=.5, random_state=0):
    X, y = data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    return {
      'confusion_matrix': confusion_matrix(y_test, y_pred),
      'accuracy_score': accuracy_score(y_test, y_pred),
      'classification_report': classification_report(y_test, y_pred),
    }