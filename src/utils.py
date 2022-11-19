# remove warnings
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import (
    feature_extraction,
    model_selection,
    naive_bayes,
    pipeline,
    manifold,
    preprocessing,
    feature_selection,
)
from sklearn.metrics import precision_recall_curve
from sklearn import metrics


def performance_report(y_test, predicted_prob, precision_threshold):

    # understand threshold to have 95% precision
    t = threshold(y_test, predicted_prob[:, 1], precision_threshold)

    print("Probability Threshold: ", t)

    # translate probabilities into predictions
    predicted = predicted_prob[:, 1] > t

    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test_array, predicted_prob, multi_class="ovr")

    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(
        xlabel="Pred",
        ylabel="True",
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion matrix",
    )
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test_array[:, i], predicted_prob[:, i]
        )
        ax[0].plot(
            fpr,
            tpr,
            lw=3,
            label="{0} (area={1:0.2f})".format(classes[i], metrics.auc(fpr, tpr)),
        )
    ax[0].plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
    ax[0].set(
        xlim=[-0.05, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate (Recall)",
        title="Receiver operating characteristic",
    )
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_array[:, i], predicted_prob[:, i]
        )
        ax[1].plot(
            recall,
            precision,
            lw=3,
            label="{0} (area={1:0.2f})".format(
                classes[i], metrics.auc(recall, precision)
            ),
        )
    ax[1].set(
        xlim=[0.0, 1.05],
        ylim=[0.0, 1.05],
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall curve",
    )
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()


def threshold(y_true, y_pred, precision_threshold):

    pre, rec, t = precision_recall_curve(y_true, y_pred)

    # get threshold that allows for >X% precision
    return t[next(x for x, val in enumerate(pre) if val >= precision_threshold)]
