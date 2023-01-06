# Import the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = np.load('cluster.npy')

# Define the function to find the closest centroids


def find_centroids(data, centroids):
    idx = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(data[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)
    return idx

# Define the function to compute and update the centroids


def compute_centroids(data, idx, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(data[idx == i], axis=0)
    return centroids

# Define the function to run the K-means algorithm


def run_k_means(data, initial_centroids, max_iters):
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_centroids(data, centroids)
        centroids = compute_centroids(data, idx, centroids.shape[0])
    return idx, centroids

# Define the function to choose random centroids


def choose_random_centroids(data, k):
    m = data.shape[0]
    idx = np.random.choice(m, k, replace=False)
    centroids = data[idx]
    return centroids

# Define the function to plot the data


def plot_k_means(data, idx, centroids):
    plt.scatter(data[:, 0], data[:, 1], c=idx, cmap='rainbow')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black')
    plt.show()


# Run the K-means algorithm
centroids = choose_random_centroids(dataset, 3)
idx, centroids = run_k_means(dataset, centroids, 10)
plot_k_means(dataset, idx, centroids)
