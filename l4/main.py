import os

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize, StandardScaler


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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = MLPClassifier(early_stopping=True)

    parameter_space = {
        #'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        #'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3, verbose=3)
    clf.fit(x_train, y_train)
    classifier.fit(x_train, y_train)

    print('Best parameters found:\n', clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    predicted = clf.predict(x_test)
    print("Accuracy test:", accuracy_score(y_test, predicted))

    print("Classification report:\n", classification_report(y_test,
                                                            predicted,
                                                            target_names=["not failed", "failed"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, predicted))