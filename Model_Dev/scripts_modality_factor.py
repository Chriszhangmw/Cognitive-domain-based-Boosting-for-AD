import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
import matplotlib.pyplot as plt


np.random.seed(42)
x = np.linspace(0.01, 0.99, 100)
true_k = 1.5
y = 0.5 * np.log((1 - x) / x) + true_k * np.exp(1 - x) + np.random.normal(0, 0.1, size=x.shape)


def calculate_entropy(data):
    hist, bin_edges = np.histogram(data, bins=10, density=True)
    return entropy(hist)

features_entropy = np.array([calculate_entropy(x)] * len(x))
features_variance = np.var(x) * np.ones_like(x)


data = pd.DataFrame({'x': x, 'y': y, 'entropy': features_entropy, 'variance': features_variance})


X = data[['entropy', 'variance']]
y_k = np.full(X.shape[0], true_k)

model = LinearRegression()
model.fit(X, y_k)
predicted_k = model.predict(X)


weights = 0.5 * np.log((1 - x) / x) + predicted_k * np.exp(1 - x)


