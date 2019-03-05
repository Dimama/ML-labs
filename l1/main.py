from process_data import get_data_from_file
from hard_drive_classifier import HardDriveClassifier
from sklearn.neighbors import KNeighborsClassifier


# TODO: refit classifier each time
def calculate_predict_time(classifier, count=10):
    import time
    sum_time = 0
    for _ in range(count):
        t1 = time.time()
        classifier.predict()
        t2 = time.time()
        sum_time += (t2 - t1)

    print("Average predict time: ", sum_time/count)


if __name__ == "__main__":
    data = get_data_from_file("harddrive1.arff")
    classifier = KNeighborsClassifier(n_jobs=-1, p=2, n_neighbors=7, weights="distance")
    #hd_classifier = HardDriveClassifier(data, classifier_file="classifier.saved", k=0.9)
    hd_classifier = HardDriveClassifier(data, classifier=classifier, k=0.9, optimize=False)
    print(hd_classifier)

    hd_classifier.fit()
    hd_classifier.predict()
    # calculate_predict_time(hd_classifier)  # 6.07, 7.83
