import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.preprocessing import normalize


def get_harddrive_data_from_file(filename):
    if not os.path.exists(filename):
        raise Exception("File {} does not exist".format(filename))

    data = {"x": [], "y": []}

    with open(filename, "r") as f:
        for line in f:
            if line[0] in ["@", "%"] or line == "\n":
                continue

            values = line.split(",")

            # ignore drive serial number
            data["x"].append([float(value) for value in values[1:-1]])
            data["y"].append(float(values[-1][:-1]))

    return data


if __name__ == "__main__":

    filename = "../l1/harddrive1.arff"
    data = get_harddrive_data_from_file(filename)

    x = np.array(normalize(data["x"]))
    y = np.array(data["y"])

    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(x)
    x = pca.transform(x)

    colors = ['red', 'green']
    for color, i in zip(colors, [0, 1]):
        ax.scatter(x[y == i, 0], x[y == i, 1], x[y == i, 2], color=color, alpha=.8, lw=3, label=i)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()