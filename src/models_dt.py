import numpy as np
import pandas as pd
from data import get_data_dropNaN
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


if __name__ == '__main__':
    df_train = pd.read_csv('../data/churn_train.csv')
    X_train, y_train = get_data_dropNaN(df_train)
    df_test = pd.read_csv('../data/churn_test.csv')
    X_test, y_test = get_data_dropNaN(df_test)

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

    # SVC model
    Cvals = [0.01, 0.1, 1.0, 10.0, 100.0]

    for c in Cvals:
        svc = SVC(C=c, kernel='linear')
        svc.fit(X_train,y_train)
        plot_svm_decision(svc, X_train.values, y_train.values, label_sizes, name)
        plt.show()
