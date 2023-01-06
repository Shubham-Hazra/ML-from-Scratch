import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

x = []
y = []

with open('lin_regr_data.txt', 'r') as f:
    for line in f.readlines():
        data = line.split(',')
        data[0] = float(data[0])
        data[1] = float(data[1])
        x.append(data[0])
        y.append(data[1])
x = np.array(x)
y = np.array(y)

# plt.scatter(x, y)
# plt.show()


def cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    cost = 0
    for i in range(m):
        f = w*x[i]+b
        cost += (f - y[i])**2
    total_cost = cost/(2*m)
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dw = 0
    db = 0
    for i in range(m):
        f = w*x[i]+b
        dw += (f-y[i])*x[i]
        db += (f-y[i])
    dw = dw/m
    db = db/m
    return dw, db


def gradient_descent(x, y, w, b, learning_rate, iterations):
    cost_history = []
    for _ in range(iterations):
        dw, db = compute_gradient(x, y, w, b)
        w = w - learning_rate*dw
        b = b - learning_rate*db
        cost_history.append(cost(x, y, w, b))
    return w, b, cost_history


w_init = 0
b_init = 0
learning_rate = 0.004
iterations = 10000

w, b, cost_history = gradient_descent(
    x, y, w_init, b_init, learning_rate, iterations)

# print("w: ", w)
# print("b: ", b)

# plt.plot(cost_history)
# plt.show()

plt.scatter(x, y)
plt.plot(x, w*x+b, color='red')
plt.show()
