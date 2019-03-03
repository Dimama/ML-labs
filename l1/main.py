from process_data import get_data_from_file
from hard_drive_classifier import HardDriveClassifier
from sklearn.neighbors import KNeighborsClassifier
#import argparse

if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description="Hard drive classifier")
    #parser.add_argument("--data", help="File with data", metavar="filename", required=True)
    #parser.add_argument("--load", help="File with trained classifier", metavar="filename", required=False)
    #parser.add_argument("--save", help="File to save classifier", metavar="filename", required=False)
    #parser.parse_args()

    data = get_data_from_file("harddrive1.arff")
    classifier = KNeighborsClassifier()
    hd_classifier = HardDriveClassifier(data, classifier_file="classifier.saved", k=0.9)
    print(hd_classifier)

    #hd_classifier.fit(save=True)
    hd_classifier.predict()



