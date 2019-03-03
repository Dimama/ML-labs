import matplotlib.pyplot as plt
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from process_data import get_data_from_file


def find_optimal_classifier_params(filename):
    data = get_data_from_file(filename)

    k = [i for i in range(3, 6) if i % 2 == 1]
    parameters = {"n_neighbors": k,
                  "p": [1, 2]}
    kn = KNeighborsClassifier(n_jobs=-1, metric="minkowski")

    clf = GridSearchCV(kn, parameters, cv=5, verbose=3)
    clf.fit(data["x"], data["y"])

    y1 = []
    y2 = []
    print(clf.cv_results_["param_p"])
    print(clf.cv_results_["mean_test_score"])

    for i, val in enumerate(clf.cv_results_["mean_test_score"]):
        if clf.cv_results_["param_p"][i] == 1:
            y1.append(val)
        else:
            y2.append(val)

    print(k, y1, y2)
    return k, y1, y2


def create_graphs(x, y1, y2, x_label, y1_label, y2_label):
    plt.title("Алгоритм k-ближайших соседей")
    plt.xlabel(x_label)
    plt.ylabel("Mean accuracy")

    y1_dots, y2_dots = plt.plot(x, y1, 'go:', x, y2, 'r^:', )

    plt.legend((y1_dots, y2_dots), (y1_label, y2_label), loc='lower right')

    plt.grid(which='minor', alpha=0.2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.show()


if __name__ == "__main__":
    k, y1, y2 = find_optimal_classifier_params("harddrive1.arff")
    create_graphs(k, y1, y2, "k", "manhattan", "euclidean")
