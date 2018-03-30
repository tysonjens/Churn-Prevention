import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# function imports
from data import *
from plotroc import plot_roc_curve
# imports
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


if __name__ == '__main__':
    df_train = pd.read_csv('../data/churn_train.csv')
    X_train, y_train = get_data(df_train)
    df_test = pd.read_csv('../data/churn_test.csv')
    X_test, y_test = get_data(df_test)

    # logistic regression
    pipe_logistic = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression())
            ])
    pipe_logistic.fit(X_train, y_train)
    print(pipe_logistic.score(X_test, y_test))

    acc = cross_val_score(pipe_logistic,X_test, y_test, cv=5, scoring='accuracy')
    precision = cross_val_score(pipe_logistic,X_test, y_test, cv=5, scoring='precision')
    print('accuracy = {}'.format(np.mean(acc)))
    print('precision = {}'.format(np.mean(precision)))

    coefs = pipe_logistic.named_steps['logistic'].coef_

    print(coefs)

    # adaboost classifier
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    predict = ada.predict(X_test)
    print('AdaBoost accuracy:', ada.score(X_test, y_test))
    acc = cross_val_score(ada,X_test, y_test, cv=5, scoring='accuracy')
    precision = cross_val_score(ada,X_test, y_test, cv=5, scoring='precision')
    print('accuracy = {}'.format(np.mean(acc)))
    print('precision = {}'.format(np.mean(precision)))

    # roc curve
    y_test_preds = pipe_logistic.named_steps['logistic'].predict_proba(X_test)[:,1]
    plot_roc_curve(y_test,y_test_preds)
    #plt.show()
