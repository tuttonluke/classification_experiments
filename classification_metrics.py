# %%
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt

# %%
def min_max_norm(X):
    sX = MinMaxScaler()

    scaled_X = sX.fit_transform(X)

    return scaled_X

def visualise_confusion_matrix(y_true, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix)
    display.plot()
    plt.show()

def plot_roc_curve(y_true, y_score):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_true, 
                                                                        y_score)
    area_under_curve = metrics.auc(false_positive_rate, true_positive_rate)
    display = metrics.RocCurveDisplay(fpr=false_positive_rate, 
                                        tpr=true_positive_rate, 
                                        roc_auc=area_under_curve,
                                        estimator_name='example estimator')
    display.plot()
    plt.show()

# %%
if __name__ == "__main__":
    np.random.seed(42)

    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train_scaled = min_max_norm(X_train)
    X_test_scaled = min_max_norm(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_test_pred = model.predict(X_test_scaled)

    visualise_confusion_matrix(y_test, y_test_pred)

    print("Accuracy:", metrics.accuracy_score(y_test, y_test_pred))
    print("Precision:", metrics.precision_score(y_test, y_test_pred, average="macro"))
    print("Recall:", metrics.recall_score(y_test, y_test_pred, average="macro"))
    print("F1 score:", metrics.f1_score(y_test, y_test_pred, average="macro"))

    y_test_score = model.decision_function(X_test_scaled)
    plot_roc_curve(y_test, y_test_score)