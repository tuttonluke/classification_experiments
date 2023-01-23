# %%
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
import numpy as np
from utils import visualise_predictions, show_data, get_classification_data
# %%
if __name__ == "__main__":
    np.random.seed(42)

    m = 60
    n_features = 2
    n_classes = 2
    X, Y = get_classification_data(sd=10, m=60, n_clusters=n_classes, n_features=n_features)
    show_data(X, Y)
    

    classification_tree = tree.DecisionTreeClassifier(max_depth=50)
    classification_tree.fit(X, Y)
    classification_tree.predict(X)

    visualise_predictions(classification_tree.predict, X)
    show_data(X, Y)
# %%
