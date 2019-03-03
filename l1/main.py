from process_data import get_data_from_file
from hard_drive_classifier import HardDriveClassifier
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":

    data = get_data_from_file("harddrive1.arff")
    classifier = KNeighborsClassifier()
    hd_classifier = HardDriveClassifier(data, classifier_file="classifier.saved", k=0.9)
    print(hd_classifier)

    #hd_classifier.fit(save=True)
    hd_classifier.predict()



