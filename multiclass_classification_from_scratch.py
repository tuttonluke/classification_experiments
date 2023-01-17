# %%
from logistic_regression_from_scratch import min_max_norm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %% code adapted from https://www.kaggle.com/code/lildatascientist/multiclass-classification-from-scratch
class LogisticRegression:
    
    def __init__(self, lr=0.1, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        #init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iters):
            linear_model = X @ self.weights + self.bias
            hx = self._sigmoid(linear_model)
            
            dw = (X.T * (hx - y)).T.mean(axis=0)
            db = (hx - y).mean(axis=0)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db 

    def predict(self,X):
        linear_model = np.dot(X,self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return y_predicted
  
    def _sigmoid(self,x):
        return(1/(1+np.exp(-x)))
# %%
class MyMulticlassClassification:
    def __init__(self) -> None:
        self.models = []
    
    def fit(self, X, y):
        for y_i in np.unique(y):
            # y_i - positive class for now
            # All other classes except y_i are negative

            # Choose x where y is positive class
            X_true = X[y == y_i]
            # Choose x where y is negative class
            X_false = X[y != y_i]
            # Concatenate
            X_true_false = np.vstack((X_true, X_false))

            # Set y to 1 where it is positive class
            y_true = np.ones(X_true.shape[0])
            # Set y to 0 where it is negative class
            y_false = np.zeros(X_false.shape[0])
            # Concatenate
            y_true_false = np.hstack((y_true, y_false))

            logit_model = LogisticRegression()
            logit_model.fit(X_true_false, y_true_false)
            self.models.append([y_i, logit_model])
        
    def predict(self, X):

        y_pred = [[label, model.predict(X)] for label, model in self.models]

        output = []

        for i in range(X.shape[0]):
            max_label = None
            max_prob = -10**5
            for j in range(len(y_pred)):
                prob = y_pred[j][1][i]
                if prob > max_prob:
                    max_label = y_pred[j][0]
                    max_prob = prob
            output.append(max_label)
        
        return output

                    

# %%
if __name__ == "__main__":
    # set random seed
    np.random.seed(42)

    # load in data
    X, y = datasets.load_iris(return_X_y=True)
    
    # split data into trian, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                    y_test, test_size=0.3)    

    # min-max normalise the training data                                                                
    X_train_scaled = min_max_norm(X_train)
    X_test_scaled = min_max_norm(X_test)
    X_validation_scaled = min_max_norm(X_validation)

    #print(X_train_scaled, "\n")
    model = MyMulticlassClassification()
    model.fit(X_train_scaled, y_train)
    model.predict(X_test_scaled)

    print(accuracy_score(y_test, model.predict(X_test_scaled)))

# %%
