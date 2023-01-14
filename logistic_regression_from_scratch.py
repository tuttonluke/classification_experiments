# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# %%
def min_max_norm(X):
    sX = MinMaxScaler()

    scaled_X = sX.fit_transform(X)

    return scaled_X
# %%
class MyLogisticRegression:
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
        """Updates weights and bias in gradient descent.
        """
        # predict y_hat using sigmoid funciton on linear regression formula WX + b
        y_hat = 1 / (1 + np.exp( - (self.X.dot(self.W) + self.b)))

        # calculate gradients for W and b
        tmp = np.reshape((y_hat - self.y.T), self.no_training_examples)
        dW = np.dot(self.X.T, tmp) / self.no_training_examples
        db = np.sum(tmp) / self.no_training_examples

        # update values for W and b
        self.W = self.W - dW * self.learning_rate
        self.b = self.b - db * self.learning_rate

        return self
    
    def predict(self, X):
        """Predict hypothetical function h(X) using sigmoid function.

        Parameters
        ----------
        X : np.array
            Array of feature values to make label predictions from.

        Returns
        -------
        np.array
            Array of binary predictions for each row.
        """
        prediction_probability = 1 / (1 + np.exp( - (X.dot(self.W) + self.b)))
        prediction_outcome = np.where(prediction_probability > 0.5, 1, 0)

        return prediction_outcome
# %% 
if __name__ == "__main__":
    # set random seed
    np.random.seed(42)

    # load in data
    X, y = datasets.load_breast_cancer(return_X_y=True)
    
    # split data into trian, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                    y_test, test_size=0.3)    

    # min-max normalise the training data                                                                
    X_train_scaled = min_max_norm(X_train)
    X_test_scaled = min_max_norm(X_test)
    X_validation_scaled = min_max_norm(X_validation)

    # model training
    model = MyLogisticRegression(learning_rate=0.01, n_epochs=1000)
    model.fit(X_train_scaled, y_train)

    sklearn_model = LogisticRegression()
    sklearn_model.fit(X_train_scaled, y_train)

    # prdictions
    y_pred_test = model.predict(X_test_scaled)
    y_pred_val = model.predict(X_validation_scaled)

    y_pred_test_sklearn = model.predict(X_test_scaled)
    y_pred_val_sklearn = model.predict(X_validation_scaled)

    # measure performance    
    correctly_classified = 0    
    correctly_classified_sklearn = 0
         
    count = 0    
    for count in range( np.size( y_pred_test ) ) :  
        
        if y_test[count] == y_pred_test[count] :            
            correctly_classified += 1
        
        if y_test[count] == y_pred_test_sklearn[count] :            
            correctly_classified_sklearn += 1
              
        count += 1
          
    print( "Accuracy on test set by our model: ", round(( 
      correctly_classified / count ) * 100 , 2))
    
    print( "Accuracy on test set by sklearn model: ", round(( 
      correctly_classified_sklearn / count ) * 100, 2 ))
# %%
