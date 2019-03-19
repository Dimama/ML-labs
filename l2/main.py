from clustering import k_means, agglomerative
from process_data import get_data_from_file


filename = "turkiye-student-evaluation_generic.csv"

if __name__ == "__main__":
    data = get_data_from_file(filename)
    x = [vec[5:] for vec in data]  # only Q1-Q28
    k_means(x)
    agglomerative(x)


