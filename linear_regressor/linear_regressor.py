import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def make_dataset(n_samples:int):
    x = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * x + np.random.randn(n_samples, 1)

    return x, y

n_samples = 100
x, y = make_dataset(n_samples)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
print(model.predict(np.array([[5], [2]])))
