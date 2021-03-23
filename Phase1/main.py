from pathlib import Path
from scipy.integrate import quad
import math
import numpy as np
import pandas as pd
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def normalization(data):
    d = []
    for j in range(20):
        d.append([float(i) for i in data[j]])

    for i in range(20):
        l = [abs(ele) for ele in d[i]]
        amax = max(l)
        for j, val in enumerate(d[i]):
            d[i][j] = val / amax

    return d


def pdf(x, sigma=0.25, mu=0.0):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))


def bands(r):
    array = np.array([0.0] * 2 * r)
    for i in range(1, 2 * r + 1):
        val1, err = quad(pdf, (i - r - 1) / r, (i - r) / r)
        val2, err = quad(pdf, -1, 1)
        length = 2 * (val1 / val2)
        array[i - 1] = length

    x = []
    left = -1
    for j in range(2 * r):
        right = left + array[j]
        x.append([left, right])
        left = right

    return x


def task1(w, s):
    p = Path("Output")
    p.mkdir(exist_ok=True)
    directory = "./Z"
    allFiles = os.listdir(directory)

    band = bands(3)[:6]
    for f in allFiles:
        if f.endswith(".csv"):
            data = pd.read_csv(directory + "/" + f, header=None).to_numpy(dtype='float64')
            completeName = os.path.join(p, "{0}.wrd.csv".format(f.split('.')[0]))
            with open(completeName, mode='w') as file:
                for index in range(20):
                    ts = normalization(data)[index]
                    length = len(ts)
                    wrd = []
                    for j in range(length):
                        for i in range(6):
                            if band[i][0] <= ts[j] <= band[i][1]:
                                wrd.append(i + 1)

                    x = []
                    writer = csv.writer(file)
                    for i in range(0, len(wrd), s):
                        vector = wrd[i:i + w]
                        if len(vector) == 3:
                            x.append('<<{0},{1},{2}>,{3},{4},{5}>'.format(f, index + 1, i, vector[0], vector[1], vector[2]))
                    writer.writerow(x)

    return x


def task2():
    directory = "./Output"
    allFiles = os.listdir(directory)

    dict = {}  # store the count of each word (string, dict)
    wrd_freq = {}  # number of word appeared in directory
    wrd_list = []  # store all words
    wrd_file = {}  # store the count of the word that appeared in files
    sensor_file = {}  # number of word appeared in each sensor with format: (string, int)

    num_files = 0

    # read files under dataset
    for file in allFiles:
        # extract the files that only came from task1
        num_files += 1
        data = pd.read_csv(directory + "/" + file, header=None).values.tolist()

        count_wrd = {}  # used to store the count of words, format: (str, int)
        sensor_dict = {"num_sensors": len(data)}

        # parser, and extract only word for each time series
        for i in range(len(data)):
            word_temp = []  # store extinct word value
            for y in range(len(data[0])):
                full_word = data[i][y].split(",")
                temp = ""

                # extract word
                for j in range(3, len(full_word)):
                    # delete last element ">" in the word
                    if j == len(full_word) - 1:
                        temp += full_word[j].split(">")[0]
                    else:
                        temp += full_word[j]

                if temp not in word_temp:
                    word_temp.append(temp)

                # calculate the max_frequency
                if wrd_freq.get(temp, -1) == -1:
                    wrd_freq[temp] = 1
                    wrd_list.append(temp)
                else:
                    wrd_freq[temp] = wrd_freq[temp] + 1

                # count the word in under this file
                if count_wrd.get(temp, -1) == -1:
                    count_wrd[temp] = 1
                else:
                    count_wrd[temp] = count_wrd[temp] + 1

                dict[file] = count_wrd

            # count number in each sensor
            for word in word_temp:
                if word in sensor_dict:
                    sensor_dict[word] += 1
                else:
                    sensor_dict[word] = 1

            sensor_file[file] = sensor_dict

    # if the word does not appear in the file, return 0:
    for k, v in dict.items():
        for temp in wrd_list:
            if v.get(temp, -1) == -1:
                v[temp] = 0

    # find number of word appear in files
    for temp in wrd_list:
        for value in dict.values():
            if value[temp] != 0:
                if temp in wrd_file:
                    wrd_file[temp] += 1
                else:
                    wrd_file[temp] = 1

    output = ""
    for word in wrd_list:
        output += word + ","
    output = output[:-1]  # format output
    output += "\n\n"

    for file, value in dict.items():
        output += file + "\n<"

        # TF
        total_word_inFile = sum(value.values())
        for word in wrd_list:
            output += str(value[word] / total_word_inFile) + ","
        output = output[:-1]  # format output
        output += ">\n<"

        # TF-IDF (0.5 + 0.5 * (tf_value / max_frequency_word_in_dir)) * idf
        for word in wrd_list:
            tf_value = value[word] / total_word_inFile
            idf = math.log10(num_files / wrd_file[word])
            tf_idf = idf * tf_value
            output += str(tf_idf) + ","
        output = output[:-1]  # format output
        output += ">\n<"

        #  TF-IDF2 (0.5 + 0.5 * (tf_value / max_frequency_word_in_file)) * idf2
        temp_dict = sensor_file[file]
        for word in wrd_list:
            if word in temp_dict:
                tf_value = value[word] / total_word_inFile
                idf2 = math.log10(temp_dict["num_sensors"] / temp_dict[word])
                tf_idf2 = tf_value * idf2
                output += str(tf_idf2) + ","
            else:
                output += str(float(0)) + ","
        output = output[:-1]
        output += ">\n\n"

    with open(directory + "/vectors.txt", 'w') as output_file:
        output_file.write(output)

    return output


def task3(f, choice):
    """
    :param dir: Given Directory
    :param f:   File selected by user
    :param choice: Choose only between TF, TF_IDF, TF_IDF2
    :return:
    """

    directory = "./Output"
    file_name = f + ".wrd.csv"

    with open(directory + "/" + file_name) as file:
        data = pd.read_csv(directory + "/" + file_name, header=None).values.tolist()
        row = len(data)
        col = len(data[0])
        word_data = np.zeros((row, col), dtype=int)

        for i in range(row):
            for y in range(col):
                full_word = data[i][y].split(",")
                temp = ""

                # extract word
                for z in range(3, len(full_word)):
                    # delete last element ">" in the word
                    if z == len(full_word) - 1:
                        temp += full_word[z].split(">")[0]
                    else:
                        temp += full_word[z]
                # print(temp)
                word_data[i][y] = int(temp)
        # print(word_data)

        # extract TF or TF_IDF or TF_IDF2 value based on user choice
        with open(directory + "/vectors.txt") as vector_file:
            all_words = vector_file.readline()[:-1].split(",")
            # print(all_words)
            temp = {}
            # print(all_words)
            for word in all_words:
                temp[word] = 0
            while True:
                nextLine = vector_file.readline()[:-1]
                if nextLine == file_name:
                    break
            if choice == "TF":
                nextLine = vector_file.readline()[1:-2]

            elif choice == "TF_IDF":
                vector_file.readline()  # flush useless values
                nextLine = vector_file.readline()[1:-2]

            elif choice == "TF_IDF2":
                vector_file.readline()
                vector_file.readline()  # flush useless values
                nextLine = vector_file.readline()[1:-2]

            else:
                print("The choice is only selected from below:\n TF, "
                      "TF_IDF, TF_IDF2")
                return
            
            # get the values of all values below the file names
            all_value = np.array(nextLine.split(","), dtype="float64")
            
            for index in range(len(all_words)):
                temp[all_words[index]] = all_value[index]
                
            # plot the heatmap
            plot_heatmap(word_data, temp, f, choice)

        return f, choice


def plot_heatmap(text, temp, f, choice):
    # text is a 2D map whose x-axis is sensor and y-axis is time series
    data = np.zeros((len(text), len(text[0])), dtype='float64')
    max_frequency = max(temp.values())

    # re-normalize to 0-255
    for row in range(len(data)):
        for col in range(len(data[0])):
            data[row][col] = (temp[str(text[row][col])] / max_frequency) * 255

    # set up environment
    plt.figure(figsize=(20, 8), dpi=80)
    sns.set(font_scale=0.7)
    t = sns.heatmap(data, cmap=cm.gray_r, annot=True, fmt=".1f", vmax=255, vmin=0)
    t.set_yticklabels(t.get_yticklabels(), rotation=0)

    # plt.show()

    plt.savefig("./ImgOutput/{0}-{1}.png".format(f, choice))

    return


def task4(f, choice):
    # users can choose which input they want to compare
    directory = "./Output"
    with open(directory + "/vectors.txt") as vector_file:
        vector_file.readline()
        print(vector_file.readline() + "")
        temp = {}
        count = 0
        file_name = vector_file.readline()[:-1]

        # loop the all files
        while file_name.endswith(".wrd.csv"):
            # do TF comparison
            if choice == "TF":
                tf_values = vector_file.readline()[1:-2].split(",")
                temp[file_name] = tf_values
                vector_file.readline()
                vector_file.readline()  # flush
            # do TF-IDF comparison
            elif choice == "TF_IDF":
                vector_file.readline()  # flush
                tf_idf_values = vector_file.readline()[1:-2].split(",")
                temp[file_name] = tf_idf_values
                vector_file.readline()  # flush
            # do TF-IDF2 comparison
            elif choice == "TF_IDF2":
                vector_file.readline()
                vector_file.readline()  # flush
                tf_idf2_values = vector_file.readline()[1:-2].split(",")
                temp[file_name] = tf_idf2_values
            else:
                print("The choice is only selected from TF, TF_IDF, and TF_IDF2")
                return
            count += 1
            vector_file.readline()
            file_name = vector_file.readline()[:-1]

        # similar gestures
        similar(f, temp)

    return f, temp


def similar(file, temp):
    similarity = {}
    file_name = file + ".wrd.csv"
    # print(temp.items())
    for file, values in temp.items():
        v1 = temp[file_name]
        # print(v1)
        numerator = 0
        denominator = 0

        # use the formula of normalization to sort the similarities
        for i in range(len(values)):
            temp1 = float(values[i])
            temp2 = float(v1[i])
            numerator += min(temp1, temp2)
            denominator += max(temp1, temp2)
        similarity[file] = numerator / denominator
    all_files = list(similarity.keys())

    top10 = sorted(list(similarity.values()), reverse=True)[0:10]
    result = []
    for file in all_files:
        if similarity[file] in top10:
            result.append(file.split(".")[0])

    # print the result into log box
    print(result)

    with open("SimilarOutput/task4output.txt", 'w') as output_file:
        output_file.write(str(result))


if __name__ == '__main__':
    files = os.listdir("./testDataForReport")
    p = Path("ImgOutput")
    p.mkdir(exist_ok=True)
    p4 = Path("SimilarOutput")
    p4.mkdir(exist_ok=True)

    task1(3, 2)

    task2()

    for i in range(1, len(files)+1):
        task3("test{0}".format(i), "TF")
    for i in range(1, len(files)+1):
        task3("test{0}".format(i), "TF_IDF")
    for i in range(1, len(files)+1):
        task3("test{0}".format(i), "TF_IDF2")

    task4("test1", "TF_IDF2")
