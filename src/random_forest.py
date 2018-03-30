def random_forest(X_train, X_test,y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    import numpy as np

    rfc = RandomForestClassifier(n_estimators=120, min_samples_leaf=4)
    rfc.fit(X_train, y_train);

    y_preds_rfc = rfc.predict_proba(X_test)[:,1]
    y_preds_rfc_bin = rfc.predict(X_test)

    #TPRrfc, FPRrfc, thresholdsrfc = roc_curve(y_test, y_preds_rfc, pos_label=None, sample_weight=None, drop_intermediate=True)

    #plotroc(TPRrfc, FPRrfc)

    rfc_prec = np.mean(cross_val_score(rfc, X_train, y_train, scoring='precision', cv=5))
    rfc_acc = np.mean(cross_val_score(rfc, X_train, y_train, scoring='accuracy', cv=5))
    rfc_test_prec = precision_score(y_test, y_preds_rfc_bin)
    rfc_test_acc = accuracy_score(y_test, y_preds_rfc_bin)
    print("The cross validated precision score is {:0.3}".format(rfc_prec))
    print("The cross validated accuracy score is {:0.3}".format(rfc_acc))
    print("The test precision score is {:0.3}".format(rfc_test_prec))
    print("The test accuracy score is {:0.3}".format(rfc_test_acc))

    return(rfc)
