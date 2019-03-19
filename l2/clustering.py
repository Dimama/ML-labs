import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from prettytable import PrettyTable


def k_means(x):
    inertia = []
    count = 10
    for k in range(2, count):
        kmeans = KMeans(n_clusters=k).fit(x)
        inertia.append(kmeans.inertia_)

    plt.plot(range(2, count), inertia, marker="o")
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


def compare_silhouettes(x, count=10):
    table = PrettyTable()
    table.field_names = ["k", "KMeans", "Agglomerative"]
    for k in range(2, count+1):
        k_means = KMeans(n_clusters=k).fit(x)
        aggl = AgglomerativeClustering(n_clusters=k).fit(x)
        table.add_row([k,
                       silhouette_score(x, k_means.labels_),
                       silhouette_score(x, aggl.labels_)])

    print("Silhouettes comparison")
    print(table)
