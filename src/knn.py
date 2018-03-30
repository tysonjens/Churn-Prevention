def knn(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    knn = KNeighborsClassifier(n_neighbors=50)

    scal = StandardScaler()
    X_train_scal = scal.fit_transform(X_train)
    X_test_scal = scal.transform(X_test)

    knn.fit(X_train_scal, y_train);

    y_preds_knn = knn.predict_proba(X_test_scal)[:,1]
    y_preds_knn_bin = knn.predict(X_test_scal)

    #TPRknn, FPRknn, thresholdsknn = roc_curve(y_test, y_preds_knn, pos_label=None, sample_weight=None, drop_intermediate=True)

    #plotroc(TPRknn, FPRknn)

    knn_prec = np.mean(cross_val_score(knn, X_train, y_train, scoring='precision', cv=5))
    knn_acc = np.mean(cross_val_score(knn, X_train, y_train, scoring='accuracy', cv=5))
    knn_test_prec = precision_score(y_test, y_preds_knn_bin)
    knn_test_acc = accuracy_score(y_test, y_preds_knn_bin)
    print("The cross validated precision score is {:0.3}".format(knn_prec))
    print("The cross validated accuracy score is {:0.3}".format(knn_acc))
    print("The test precision score is {:0.3}".format(knn_test_prec))
    print("The test accuracy score is {:0.3}".format(knn_test_acc))

    return(knn)
