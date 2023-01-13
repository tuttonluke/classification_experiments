# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# %%
def min_max_norm(X, y):
    sX = MinMaxScaler()
    sy = MinMaxScaler()

    scaled_X = sX.fit_transform(X)
    scaled_y = sy.fit_transform(y.reshape(-1, 1))

    return scaled_X, scaled_y
# %%
class LogitsticRegression:
    def __init__(self, learning_rate, n_epochs) -> None:
        self.learning_rate = learning_rate
        self.epochs = n_epochs

    def fit(self, X, y):
        """Function for training the model.

        Parameters
        ----------
        X : np.array
            Matrix of feature data.
        y : np.array
            Vector of label data.
        """
        self.no_training_examples, self.no_features = X.shape
        # initialise weights and bias
        self.W = np.zeros(self.no_features)
        self.b = 0
        self.X = X
        self.y = y

        # gradient descent optimisation
        for i in range(self.epochs):
            self.update_weights()
        
        return self

    def update_weights(self):
        pass
# %% 
if __name__ == "__main__":
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                    y_test, test_size=0.3)    
    X_train_scaled, y_train_scaled = min_max_norm(X_train, y_train)


    # TEST WITH SKLEARN
    np.random.seed(42)
    log_model = LogisticRegression(max_iter=10000)
    log_model.fit(X_train, y_train)
    y_pred_val = log_model.predict(X_validation)
    
# %%
