def plot_roc_curve(y_test,y_test_preds):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    TPR, FPR, thresholds = roc_curve(y_test, y_test_preds, pos_label=None, sample_weight=None, drop_intermediate=True)

    def plotroc(TPR, FPR):
        roc_auc = auc(TPR, FPR)
        plt.figure()
        lw = 2
        plt.plot(TPR, FPR, color='darkorange',
                 lw=lw, label="ROC curve area = {0:0.4f}".format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    plotroc(TPR, FPR)
