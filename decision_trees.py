# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Making the dataset
X_train = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [
                   0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0]])
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
node_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Function to compute entropy


def compute_entropy(y):
    entropy = 0.
    if len(y) != 0:
        p1 = len(y[y == 1]) / len(y)
        if p1 != 0 and p1 != 1:
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        else:
            entropy = 0
    return entropy

# Function to split the dataset at each node


def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

# Function to compute information gain


def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    information_gain = 0
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    information_gain = node_entropy - weighted_entropy
    return information_gain

# Function to get the best split with maximum information gain


def get_best_split(X, y, node_indices):
    num_features = X.shape[1]
    best_feature = -1
    max_info_gain = 0
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
    return best_feature


# List to store the tree
tree = []

# Function to build the decision tree


def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" %
              branch_name, node_indices)
        return
    best_feature = get_best_split(X, y, node_indices)
    tree.append((current_depth, branch_name, best_feature, node_indices))
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" %
          (formatting, current_depth, branch_name, best_feature))
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    build_tree_recursive(X, y, left_indices, "Left",
                         max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right",
                         max_depth, current_depth+1)


# Building the decision tree
build_tree_recursive(X_train, y_train, node_indices, "Root", 3, 0)
