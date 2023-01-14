# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
np.random.seed(42)
random_features = np.random.rand(50)
labels = np.zeros(50)

cut_off = np.random.choice(random_features)
print(cut_off)

for index, label in enumerate(labels):
    if random_features[index] > cut_off:
        labels[index] = 1

# %%
plt.figure()
plt.scatter(random_features, labels)
plt.show()