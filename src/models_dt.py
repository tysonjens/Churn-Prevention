import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# function imports
from data import *
from plotroc import plot_roc_curve
from bagging import bagging
# imports
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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

    # decision tree classifier
    dtc = DecisionTreeClassifier(max_depth=3,)
    dtc.fit(X_train, y_train);

    y_preds_dtc = dtc.predict_proba(X_test)[:,1]
    y_preds_dtc_bin = dtc.predict(X_test)

    dtc_prec = np.mean(cross_val_score(dtc, X_train, y_train, scoring='precision', cv=5))
    dtc_acc = np.mean(cross_val_score(dtc, X_train, y_train, scoring='accuracy', cv=5))
    dtc_test_prec = precision_score(y_test, y_preds_dtc_bin)
    dtc_test_acc = accuracy_score(y_test, y_preds_dtc_bin)
    print("The cross validated precision score is {:0.3}".format(dtc_prec))
    print("The cross validated accuracy score is {:0.3}".format(dtc_acc))
    print("The test precision score is {:0.3}".format(dtc_test_prec))
    print("The test accuracy score is {:0.3}".format(dtc_test_acc))

    # bagging classifier
    bagc = bagging(X_train, y_train, X_test, y_test)

    # adaboost classifier
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    predict = ada.predict(X_test)
    print('AdaBoost accuracy:', ada.score(X_test, y_test))
    acc = cross_val_score(ada,X_test, y_test, cv=5, scoring='accuracy')
    precision = cross_val_score(ada,X_test, y_test, cv=5, scoring='precision')
    print('accuracy = {}'.format(np.mean(acc)))
    print('precision = {}'.format(np.mean(precision)))

    # gradient boost classifier
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_hat = gbc.predict(X_test)
    score_accuracy = accuracy_score(y_test, y_hat)
    score_precision = precision_score(y_test, y_hat)
    print('accuracy = {}'.format(score_accuracy))
    print('precision = {}'.format(score_precision))

    # gradient boost grid search
    #%% grid search
    '''grid search to find best params for gradient boost classifier '''
    gbc_grid = {'learning_rate': np.linspace(0.2,0.8,4),
                     'max_depth': [1,2,4,8],
                     'min_samples_leaf': [2, 4, 6, 8],
                     'max_features': ['sqrt', 'log2', None],
                     'n_estimators': [100, 150, 200, 250, 300]}

    gbc_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                        gbc_grid,
                        n_jobs=-1,
                        verbose=True,
                        scoring='precision')
    #svc_gridsearch.fit(X_train, y_train)
    #print("best gbc parameters:", svc_gridsearch.best_params_)
    # # roc curve for logistic
    # y_test_preds = pipe_logistic.named_steps['logistic'].predict_proba(X_test)[:,1]
    # plot_roc_curve(y_test,y_test_preds)

    # # roc curve for adaboost
    # y_test_preds = ada.predict_proba(X_test)[:,1]
    # plot_roc_curve(y_test,y_test_preds)

    # plot muliple models on roc curve
    logistic_mod = pipe_logistic.named_steps['logistic']
    models = [logistic_mod, dtc, bagc, ada, gbc]
    model_names = ['logistic', 'decision tree', 'bagging', 'AdaBoost', 'Gradient Boost']
    colors = ['b','k','m','g','r','c']
    plt.figure()
    for idx, model in enumerate(models):
        modname = model_names[idx]
        color = colors[idx]
        y_test_preds = model.predict_proba(X_test)[:,1]
        plot_roc_curve(y_test,y_test_preds,modname,color)
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
