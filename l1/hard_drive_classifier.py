from random import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.externals import joblib


class HardDriveClassifier:
    def __init__(self, data, classifier=None, k=0.8, classifier_file=None):

        if classifier is not None:
            self.classifier = classifier
            self.is_fitted = False
        else:
            self.classifier = HardDriveClassifier._load_classifier(classifier_file)
            self.is_fitted = True

        self._split_data(data, k)
        self._get_additional_info(data)

    def fit(self, save=False):
        self.classifier.fit(self.x_train, self.y_train)
        self.is_fitted = True

        if save:
            self._save_to_file("classifier.saved")

    def predict(self):
        if not self.is_fitted:
            raise Exception("Classifies is not fitted yet!")

        predicted = self.classifier.predict(self.x_train)
        print("Accuracy train:", accuracy_score(self.y_train, predicted))

        predicted = self.classifier.predict(self.x_test)
        print("Accuracy test:", accuracy_score(self.y_test, predicted))

        print("Classification report:\n", classification_report(self.y_test,
                                                                predicted,
                                                                target_names=["not failed", "failed"]))
        print("Confusion matrix:\n", confusion_matrix(self.y_test, predicted))

    def test(self, k=None):
        pass

    def _save_to_file(self, filename):
        if not self.is_fitted:
            raise Exception("Classifies is not fitted yet!")

        joblib.dump(self.classifier, filename)
        print("Classifier saved to {}".format(filename))

    def _split_data(self, data, k):
        length = len(data["y"])
        index = int(length * k)
        indices = list(range(length))
        shuffle(indices)

        x = [data["x"][i] for i in indices]
        y = [data["y"][i] for i in indices]

        self.x_train, self.x_test = x[:index], x[index:]
        self.y_train, self.y_test = y[:index], y[index:]

    @staticmethod
    def _load_classifier(filename):
        try:
            return joblib.load(filename)
        except FileNotFoundError:
            raise Exception("File {} not found".format(filename))

    def _get_additional_info(self, data):
        self.p_count = data["y"].count(1)
        self.n_count = data["y"].count(0)

    def __repr__(self):
        return "Classifier:\n{0}\ndataset params:\n len: {1}\n 0/1 - {2}/{3}".format(self.classifier,
                                                                                     self.n_count + self.p_count,
                                                                                     self.n_count,
                                                                                     self.p_count)
