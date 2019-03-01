import os


def get_data_from_file(filename):
    """
    function to read data from *.arff file
    :param filename:
    :return:
    """
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
