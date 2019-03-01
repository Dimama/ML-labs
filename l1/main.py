from process_data import get_data_from_file
from hard_drive_classifier import HardDriveClassifier
from sklearn import svm

if __name__ == "__main__":

    data = get_data_from_file("harddrive1.arff")
    classifier = svm.SVC()
    hd_classifier = HardDriveClassifier(classifier, data)
    print(hd_classifier)
