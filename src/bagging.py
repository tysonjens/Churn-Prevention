def bagging(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    import numpy as np

    bagc = BaggingClassifier(n_estimators=100)
    bagc.fit(X_train, y_train);

    y_preds_bagc = bagc.predict_proba(X_test)[:,1]
    y_preds_bagc_bin = bagc.predict(X_test)

    #TPRbagc, FPRbagc, thresholdsbagc = roc_curve(y_test, y_preds_bagc, pos_label=None, sample_weight=None, drop_intermediate=True)

    #plotroc(TPRbagc, FPRbagc)

    bagc_prec = np.mean(cross_val_score(bagc, X_train, y_train, scoring='precision', cv=5))
    bagc_acc = np.mean(cross_val_score(bagc, X_train, y_train, scoring='accuracy', cv=5))
    bagc_test_prec = precision_score(y_test, y_preds_bagc_bin)
    bagc_test_acc = accuracy_score(y_test, y_preds_bagc_bin)
    print("The cross validated precision score is {:0.3}".format(bagc_prec))
    print("The cross validated accuracy score is {:0.3}".format(bagc_acc))
    print("The test precision score is {:0.3}".format(bagc_test_prec))
    print("The test accuracy score is {:0.3}".format(bagc_test_acc))

    return(bagc)
