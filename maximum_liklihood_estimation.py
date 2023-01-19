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
def gaussian_max_likelihood_estimation(X, mean, sigma):
    """Calculates the central position (mean) of the Gaussian distribution that best describes 
    the feature values (X) of the data, assuming that they are normally distributed.

    Parameters
    ----------
    X : np.array
        Array of feature values.
    mean : float
        Observed mean of features.
    sigma : float
        Observed standard deviation of features.

    Returns
    -------
    best_val : float
        Value of the peak of the Gaussian distribution that best describes the feature values (X).
    best_mean : float
        Mean of the Gaussian distribution that best describes the feature values (X).
    potential_mean_values : np.array
        Array of mean values tested.
    objectives : np.array
        Array of peak values of the tested means.
    """
    potential_mean_values = np.linspace(min(X), max(X), 50)
    objectives = np.zeros_like(potential_mean_values)
    best_mean = 0
    best_val = -float("inf") # initialise the best value as infinitely bad

    for index, mean in enumerate(potential_mean_values):
        gaussian = GaussianPDF(mean, sigma) # initialise a Gaussian
        objective = 0  # initialise the objective as zero

        for x in X:
            objective += np.log(gaussian(x)) # compute the log-likelihood for this example, 
                                             # and add it to the objective that we wish to maximise
        objectives[index] = objective
        if objective > best_val:
            best_val = objective
            best_mean = mean

    return best_val, best_mean, potential_mean_values, objectives

# %%
if __name__ == "__main__":
    np.random.seed(42)

    # -------- UNSUPERVISED DATA ----------

    # load data and get basic statistics
    X, y = get_regression_data()
    X = X[:, 0]
    mu = np.mean(X)
    sigma = np.std(X)
    print('Mean:', mu)
    print('Standard Deviation:', sigma)

    # initialise GaussianPDF class
    p = GaussianPDF(mu, sigma)

    domain = np.linspace(min(X) - 1, max(X) + 1)
    plt.plot(domain, p(domain))
    plt.ylim(0, 1)
    plt.xlim(min(domain), max(domain))
    plt.scatter(X, np.zeros(X.shape[0]), c="r")
    plt.show()

    best_val, best_mean, mean_values, objectives = gaussian_max_likelihood_estimation(X, mu, sigma)
    
    print('BEST VAL:', best_val)
    print('BEST MU:', best_mean)

    plt.plot(mean_values, objectives)
    plt.show()
    
# %%
