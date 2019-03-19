import os


def get_data_from_file(filename):
    if not os.path.exists(filename):
        raise Exception("File {} does not exist".format(filename))

    data = []
    with open(filename, "r") as f:
        for line in f.readlines()[1:]:
            data.append(list(map(int, line[:-1].split(","))))
    return data
