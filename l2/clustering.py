from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy


def k_means(x):
    inertia = []
    count = 10
    for k in range(1, count):
        kmeans = KMeans(n_clusters=k).fit(x)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, count), inertia, marker="o")
    plt.xlabel("$k$")
    plt.ylabel("$Inertia$")
    plt.grid(which='minor', alpha=0.2)
    plt.grid(True, linestyle='-', color='0.8')
    plt.show()


def agglomerative(x):
    hierarchy.dendrogram(hierarchy.linkage(x, method='ward'),
                         no_labels=True)
    plt.title('Dendrogram')
    plt.xlabel('questions')
    plt.ylabel('Euclidean distances')
    plt.show()


