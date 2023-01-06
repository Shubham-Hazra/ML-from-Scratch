import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

x = []
y = []

with open('log_regr_data.txt', 'r') as f:
    for line in f.readlines():
        data = line.split(',')
        data[0] = np.array([float(data[0]), float(data[1])])
        data[1] = float(data[2])
        x.append(data[0])
        y.append(data[1])
x = np.array(x)
y = np.array(y)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost(x, y, w, b):
    m, _ = x.shape
    cost = 0
    for i in range(m):
        z = np.dot(x[i], w) + b
        f_wb = sigmoid(z)
        cost += -y[i]*np.log(f_wb) - (1-y[i])*np.log(1-f_wb)
    total_cost = cost/m
    return total_cost


def compute_gradients(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * x[i, j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_db, dj_dw


def gradient_descent(x, y, w, b, learning_rate, iterations):
    cost_history = []
    for i in range(iterations):
        dj_db, dj_dw = compute_gradients(x, y, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        cost_ = cost(x, y, w, b)
        cost_history.append(cost_)
    return w, b, cost_history


w_init = 0.01 * (np.random.rand(2).reshape(-1, 1) - 0.5)
b_init = np.random.randn(1)
iterations = 100000
learning_rate = 0.001

w, b, cost_history = gradient_descent(
    x, y, w_init, b_init, learning_rate, iterations)

plt.plot(np.linspace(0, iterations, iterations), cost_history)
plt.show()


def classify(x, w, b):
    z = np.dot(x, w) + b
    f = sigmoid(z)
    if f >= 0.5:
        return 1
    else:
        return 0


predicted = []

for i in range(x.shape[0]):
    predicted.append(classify(x[i], w, b))

predicted = np.array(predicted)

print(predicted.sum())

print(x.shape)
print(y.shape)
print(predicted.shape)
plt.scatter(x[:, 0], x[:, 1], c=predicted, label='predicted')
# plt.scatter(x[:, 0], x[:, 1], c=y, label='actual')
plt.legend()
plt.show()

accuracy = np.sum(predicted == y)/y.shape[0]*(100)

print('Accuracy: ', accuracy)
