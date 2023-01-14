# %%
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression_from_scratch import MyLogisticRegression
from sklearn.model_selection import train_test_split

# %%
if __name__ == "__main__":
    np.random.seed(24)
    random_features = np.random.rand(500)
    labels = np.zeros(500)

    cut_off = np.random.choice(random_features)
    print(cut_off)

    for index, label in enumerate(labels):
        if random_features[index] > cut_off:
            labels[index] = 1
    
    X_train, X_test, y_train, y_test  = train_test_split(random_features,
                                                            labels)

    plt.figure()
    plt.scatter(random_features, labels)
    plt.xlabel("Features")
    plt.ylabel("Labels")
    plt.show()

    model = MyLogisticRegression(learning_rate=0.01, n_epochs=1000)
    model.fit(X_train.reshape(-1, 1), y_train)

    



# %%
