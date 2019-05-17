import os

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
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

    classifier = MLPClassifier(solver="lbfgs", alpha=1e-5, random_state=1)
    classifier.fit(x_train, y_train)

    predicted = classifier.predict(x_test)
    print("Accuracy test:", accuracy_score(y_test, predicted))

    print("Classification report:\n", classification_report(y_test,
                                                            predicted,
                                                            target_names=["not failed", "failed"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, predicted))