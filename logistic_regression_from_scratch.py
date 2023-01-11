# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# %%
class MyLogisticRegression:
    def __init__(self, n_features) -> None:
        self.W = np.random.randn(n_features)
        self.b = np.random.randn()

    def fit(self, X_train, y_train):
        # get W and b from linear regression
        self.linear_regression(X_train, y_train)        

    def predict(self, X):
        # apply sigmoid function to get probability
        prediction = self.sigmoid(X)
        return prediction

    def sigmoid(self, num):
        p = 1 / (1 + np.power(np.e, -num))
        return p
    
    def binary_cross_entropy(self):
        pass
    
    def linear_regression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.W = model.coef_
        self.b = model.intercept_

# %%
if __name__ == "__main__":
    np.random.seed(42)
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                    y_test, test_size=0.3)

    model = MyLogisticRegression(n_features=30)
    model.fit(X_train, y_train)
    my_y_pred_val = model.predict(X_validation)

# %% TEST WITH SKLEARN
np.random.seed(42)
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train, y_train)
y_pred_val = model.predict(X_validation)