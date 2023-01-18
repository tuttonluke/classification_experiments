# %%
import numpy as np
import matplotlib.pyplot as plt
from utils import get_regression_data

# %% MODEL WITH A GAUSSIAN DISTRIBUTION
class GaussianPDF:
    def __init__(self, mu=0, sigma=1) -> None:
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, x):
        p_x = np.exp(-(x - self.mu)**2 / (2*self.sigma**2)) / (np.sqrt(2*np.pi) * self.sigma)
        return p_x

# %%
if __name__ == "__main__":
    X, y = get_regression_data()
    X = X[:, 0]
    mu = np.mean(X)
    sigma = np.std(X)
    print('Mean:', mu)
    print('Standard Deviation:', sigma)

    p = GaussianPDF(mu, sigma)

    domain = np.linspace(min(X) - 1, max(X) + 1)
    plt.plot(domain, p(domain))
    plt.ylim(0, 1)
    plt.xlim(min(domain), max(domain))
    plt.scatter(X, np.zeros(X.shape[0]), c="r")

    plt.show()
   

    # plt.scatter(X, np.zeros(X.shape[0]), c="r")
    # plt.ylim(0, 0.3)
    # plt.show()

    # print('mean:', mu)
    # print('standard deviation:', sigma)

# %%
