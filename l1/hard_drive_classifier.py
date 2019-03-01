class HardDriveClassifier:
    def __init__(self, classifier, data, k=0.8):
        self.classifier = classifier
        self.data = data
        self.k = k
        self._get_additional_info()

    def fit(self):
        pass

    def predict(self):
        pass

    def test(self, k=None):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def _get_additional_info(self):
        self.p_count = self.data["y"].count(1)
        self.n_count = self.data["y"].count(0)

    def __repr__(self):
        return "Classifier:\n{0}\ndataset params:\n len: {1}\n 0/1 - {2}/{3}".format(self.classifier,
                                                                                     len(self.data["x"]),
                                                                                     self.n_count,
                                                                                     self.p_count)