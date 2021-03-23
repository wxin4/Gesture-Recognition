import collections
import re
import secrets
import shutil

import pandas as pd
import numpy as np
import math
import os
import csv
from pathlib import Path

import scipy
from scipy.integrate import quad
import xlrd
import matplotlib.pyplot as plt
import phase3_decisiontree


def gaussian_Integration(x, miu, sigma):
    """
    :param x:  Lower value
    :param miu: Mean value
    :param sigma:   Standard deviation
    :return: Gaussian function
    """

    return math.exp(-math.pow((x - miu), 2) / (2 * math.pow(sigma, 2)))


def length_band(r):
    """
    :param r:   Resolution
    :return:    List of length_band within a range -1 to 1
    """
    band = np.zeros(2 * r, dtype='float64')
    for i in range(2 * r):
        numerator = quad(gaussian_Integration, (i - r) / r, (i + 1 - r) / r, args=(0, 0.25))
        denominator = quad(gaussian_Integration, -1, 1, args=(0, 0.25))
        if i == 0:
            band[i] = -1 + 2 * (numerator[0] / denominator[0])
        else:
            band[i] = band[i - 1] + 2 * (numerator[0] / denominator[0])

    return band


def task0a(gestures_folder, w, s, r):
    """

    :param gestures_folder:     Given gestures folder
    :param w:       Window length
    :param s:       Shift length
    :param r:       resolution
    :return:
    """
    print("\nPerforming task 0a")
    band_Length = length_band(r)
    # find the mid point of each band
    middle_point_band = np.zeros(len(band_Length), dtype='float64')
    for i in range(len(band_Length)):
        if i == 0:
            middle_point_band[i] = (-1.0 + band_Length[i]) / 2
        else:
            middle_point_band[i] = (band_Length[i - 1] + band_Length[i]) / 2
    components = ["X", "Y", "Z", "W"]
    for cl in components:
        subfolder = gestures_folder + "/" + cl
        allFiles = os.listdir(subfolder)
        for file in allFiles:
            if (not file.endswith(".wrd.csv")) and file.endswith(".csv"):
                allData = pd.read_csv(subfolder + "/" + file, header=None).to_numpy(dtype='float64')
                row_num = len(allData)
                col_num = len(allData[0])

                output_list = [["ComponentID:", str(cl)]]  # output componentID  i
                header = ["Sensor_id", "average amplitude", "standard deviations"]
                output_list.append(header)
                for sensor_id in range(row_num):
                    output_row = []  # store output for each row
                    avg_i_j = sum(allData[sensor_id]) / col_num  # average amplitude
                    sum_SD = 0  # used to calculate standard deviation
                    max_data_in_sensor = max(allData[sensor_id]) - min(allData[sensor_id])
                    min_data_in_sensor = min(allData[sensor_id])

                    # store normalized value
                    normalized_values = np.zeros(col_num, dtype='float64')
                    # store quantized value
                    quantized_values = np.zeros(col_num, dtype='int')

                    for col_index in range(col_num):
                        sum_SD += math.pow(((allData[sensor_id][col_index]) - avg_i_j), 2)
                        if max_data_in_sensor != 0.0:
                            temp_value = 2 * (
                                    (allData[sensor_id][col_index] - min_data_in_sensor) / max_data_in_sensor) - 1
                            normalized_values[col_index] = temp_value
                        # update quantized Value
                        for y in range(2 * r):
                            if y == 0 and -1.0 <= normalized_values[col_index] <= band_Length[y]:
                                quantized_values[col_index] = y + 1
                                break
                            elif y > 0 and band_Length[y - 1] <= normalized_values[col_index] <= band_Length[y]:
                                quantized_values[col_index] = y + 1
                                break
                            elif y == (2 * r - 1) and normalized_values[col_index] <= 1.0:
                                quantized_values[col_index] = y + 1
                                break
                    std_i_j = math.sqrt(sum_SD / col_num)
                    output_row.append(str(sensor_id + 1))  # output sensor id      A
                    output_row.append(str(avg_i_j))  # output average amplitude      B
                    output_row.append(str(std_i_j))  # output standard deviations    C

                    t = 0
                    while t < col_num and (t + w) <= col_num:
                        if sensor_id == 0:
                            output_list[1].append("avgQ_" + str(t))
                            output_list[1].append("winQ_" + str(t))
                        temp_sum = 0
                        for i in range(t, t + w):
                            temp_sum += middle_point_band[quantized_values[i] - 1]
                        output_row.append(str(temp_sum / w))
                        win_str = "<"
                        for i in range(t, t + w):
                            win_str += str(quantized_values[i])
                            if i == (t + w - 1):
                                win_str += ">"
                        output_row.append(win_str)
                        t += s
                    output_list.append(output_row)

                # write to file
                output_file_name = file.split('.')[0] + ".wrd.csv"
                with open(subfolder + "/" + output_file_name, 'w') as output_file:
                    temp_writer = csv.writer(output_file)
                    for each_row in output_list:
                        temp_writer.writerow(each_row)

    return  # done for task 0a


def task0b(gestures_folder):
    """

    :param gestures_folder:     Given gestures folder
    :return:
    """
    print("\nPerforming task 0b")
    components = ["X", "Y", "Z", "W"]
    list_word = []  # store all unique tuple
    dic = {}  # general dictionary
    idf_dic = {}
    total_number_gesture_file = 0

    subfolder = gestures_folder + "/X"
    allFiles = os.listdir(subfolder)
    for file in allFiles:
        if file.endswith(".wrd.csv"):
            total_number_gesture_file += 1
            fileDic = {}
            for cl in components:
                allData = pd.read_csv(gestures_folder + "/" + cl + "/" + file, header=None, sep='\n').values.tolist()
                # parser of data
                for index, row in enumerate(allData):
                    allData[index] = row[0].split(",")
                row_num = len(allData)
                col_num = len(allData[2])
                # print(allData)
                for row_index in range(2, row_num):
                    sensor_id = allData[row_index][0]
                    h = 4
                    while h < col_num:
                        temp_word = "(" + cl + ", " + sensor_id + ", " + allData[row_index][h] + ")"
                        # update dictionary
                        if temp_word not in list_word:
                            list_word.append(temp_word)
                        # update n for tf count
                        if temp_word not in fileDic:
                            fileDic[temp_word] = 1
                            # update n for idf count
                            if temp_word not in idf_dic:
                                idf_dic[temp_word] = 1
                            else:
                                idf_dic[temp_word] += 1
                        else:
                            fileDic[temp_word] += 1
                        h += 2
            dic[file] = fileDic

    output_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for file_name, value in dic.items():
        temp = file_name.split(".")
        word_String = ""
        tf_String = ""
        tf_idf_String = ""
        total_count = sum(value.values())
        for word in list_word:
            word_String += word + "|"
            if word in value:
                tf_String += str(value[word] / total_count) + ","
                tf_idf_String += str(
                    (value[word] / total_count) * math.log10(total_number_gesture_file / idf_dic[word])) + ","
            else:
                tf_String += str(0.0) + ","
                tf_idf_String += str(0.0) + ","
        word_String = word_String[:-1]  # format output
        tf_String = tf_String[:-1]
        tf_idf_String = tf_idf_String[:-1]
        word_String += "\n\n<"
        tf_String += ">\n"
        tf_idf_String += ">\n"
        tf_outputString = word_String + tf_String
        filename = output_folder + "/tf-vectors-" + temp[0] + ".txt"
        with open(filename, 'w') as output_file:
            output_file.write(tf_outputString)

        tf_idf_outputString = word_String + tf_idf_String
        filename = output_folder + "/tfidf-vectors-" + temp[0] + ".txt"
        with open(filename, 'w') as output_file:
            output_file.write(tf_idf_outputString)

    return  # done for task 0b


def dataMatrix(output_vector_folder, vector_model):
    """

    :param output_vector_folder:    The path to vector files
    :param vector_model:        TF or TF_IDF
    :return:    list of word        unique dictionary in database
                list of files       all gesture files in database
                a data matrix       num_obj * num_words
    """
    allFiles = os.listdir(output_vector_folder)
    filename_prefix = ""
    if vector_model == "1":
        filename_prefix += "tf-vectors-"
    else:
        filename_prefix += "tfidf-vectors-"
    list_words = []
    list_files = []
    allData = []
    for file in allFiles:
        if file.startswith(filename_prefix):
            list_files.append(file)
            f = open(output_vector_folder + "/" + file, 'r')
            if len(list_words) == 0:
                list_words = f.readline()[:-1].split("|")
            else:
                f.readline()
            f.readline()  # flush a line
            data = f.readline()[1:-2].split(",")
            allData.append(data)

    allData = np.asarray(allData, dtype='float64')

    return list_words, list_files, allData


def DTW_Score(time_series_1, time_series_2):
    """

    :param time_series_1:   The first time series   with size n
    :param time_series_2:   The second time series   with size m
    :return:   bottom right score
    """
    DTW_matrix = np.zeros((len(time_series_1) + 1, len(time_series_2) + 1), dtype='float64')
    # initialization of matrix with insert basic value into matrix
    for i in range(len(DTW_matrix)):
        for y in range(len(DTW_matrix[0])):
            if i == 0 and y != 0:
                DTW_matrix[i][y] = time_series_2[y - 1]
            if y == 0 and i != 0:
                DTW_matrix[i][y] = time_series_1[i - 1]
    # compute each value
    for i in range(len(DTW_matrix)):  # index for time_series_2
        for y in range(len(DTW_matrix[0])):  # index for time_series_1
            if i != 0 and y != 0:
                DTW_matrix[i][y] = abs(time_series_2[y - 1] - time_series_1[i - 1]) + \
                                   min(DTW_matrix[i - 1][y], DTW_matrix[i - 1][y - 1], DTW_matrix[i][y - 1])
    return DTW_matrix[len(time_series_1)][len(time_series_2)]


def convert_distance_to_similarity(distance_matrix):
    """

    :param distance_matrix:     Given distance_matrix
    :return:    a similarity matrix using normalization method
    """
    similarity_matrix = []
    for i in range(len(distance_matrix)):
        normalized_value = np.zeros(len(distance_matrix[i]), dtype="float64")
        max_num = max(distance_matrix[i])
        min_num = min(distance_matrix[i])
        for y in range(len(distance_matrix[i])):
            normalized_value[y] = 1 - ((distance_matrix[i][y] - min_num) / (max_num - min_num))
        similarity_matrix.append(normalized_value)
    return np.asarray(similarity_matrix)


def generate_gesture_matrix(list_files, gestures_folder):
    """

    :param user_option:     1,2,3,4,5,6,7
    :param user_option:     gesture-feature matrix
    :param list_files:      list of all files in database
    :param gestures_folder:  Given gestures database
    :return:
    """
    list_files = [filename.split(".")[0].split("-")[2] for filename in list_files]
    num_files_in_database = len(list_files)
    DTW_distance_matrix = np.zeros((num_files_in_database, num_files_in_database), dtype="float64")
    components = ["X", "Y", "Z", "W"]
    file_data = {}  # utilization of memorization in order to decrease time complexity
    scores_dic = {}  # utilization of memorization in order to decrease time complexity
    for first_index, first_file in enumerate(list_files):
        print("Generating for gesture_" + first_file)
        if first_file not in file_data:  # extract data
            first_file_raw_data = []
            for component in components:
                allData = pd.read_csv(gestures_folder + "/" + component + "/" + first_file + ".wrd.csv",
                                      header=None,
                                      sep='\n').values.tolist()
                # parser of data
                for index, row in enumerate(allData):
                    allData[index] = row[0].split(",")
                row_num = len(allData)
                col_num = len(allData[2])

                # set size and create a numpy 2-d array
                num_row = int(row_num - 2)
                num_col = int((col_num - 3) / 2)
                avg_component_data = np.zeros((num_row, num_col), dtype='float64')
                for row_index in range(2, row_num):
                    for col_index in range(num_col):
                        h = col_index * 2 + 3
                        avg_component_data[row_index - 2][col_index] = float(allData[row_index][h])
                first_file_raw_data.append(avg_component_data)
            file_data[first_file] = first_file_raw_data  # put in dictionary
        for second_index, second_file in enumerate(list_files):
            if second_file not in file_data:
                second_file_raw_data = []
                for component in components:
                    allData = pd.read_csv(gestures_folder + "/" + component + "/" + second_file + ".wrd.csv",
                                          header=None,
                                          sep='\n').values.tolist()
                    # parser of data
                    for index, row in enumerate(allData):
                        allData[index] = row[0].split(",")
                    row_num = len(allData)
                    col_num = len(allData[2])

                    # set size and create a numpy 2-d array
                    num_row = int(row_num - 2)
                    num_col = int((col_num - 3) / 2)
                    avg_component_data = np.zeros((num_row, num_col), dtype='float64')
                    for row_index in range(2, row_num):
                        for col_index in range(num_col):
                            h = col_index * 2 + 3
                            avg_component_data[row_index - 2][col_index] = float(allData[row_index][h])
                    second_file_raw_data.append(avg_component_data)
                file_data[second_file] = second_file_raw_data

            # compute distance
            first_file_data = file_data[first_file]
            second_file_data = file_data[second_file]
            temp_compare_string1 = first_file + ", " + second_file
            temp_compare_string2 = second_file + ", " + first_file

            # utilized memorization
            if temp_compare_string1 not in scores_dic and temp_compare_string2 not in scores_dic:
                score = 0
                for component_index in range(4):  # 4 components
                    for sensor_index in range(20):  # 20 sensors
                        score += DTW_Score(first_file_data[component_index][sensor_index],
                                           second_file_data[component_index][sensor_index])
                scores_dic[temp_compare_string1] = score
                scores_dic[temp_compare_string2] = score
            DTW_distance_matrix[first_index][second_index] = scores_dic[temp_compare_string1]

    # convert to similarity matrix by using normalization of range 0-1
    gestures_similarity_matrix = convert_distance_to_similarity(DTW_distance_matrix)

    print(gestures_similarity_matrix)
    return gestures_similarity_matrix


def write_to_file(gestures_folder, vector_model, k):
    """
    :param gestures_folder:     Given gestures database
    :param vector_model:        TF or TFIDF: 1  2
    :param k:                   top k-dimension
    :return:
    """
    print("Performing task 1")
    output_vector_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_vector_folder):
        print("execute task0b first")
        return
    list_words, list_files, raw_data = dataMatrix(output_vector_folder, vector_model)

    gestures_similarity_matrix = generate_gesture_matrix(list_files, gestures_folder)
    # gestures_similarity_matrix[::-1].sort(axis=1)
    # output_matrix = gestures_similarity_matrix[:k]

    n = len(list_files)
    list_files = [filename.split(".")[0].split("-")[2] for filename in list_files]
    principalDf = pd.DataFrame(gestures_similarity_matrix, columns=list_files)
    sort_decr2_topn = lambda row, nlargest=n: sorted(pd.Series(zip(principalDf.columns, row)),
                                                     key=lambda cv: -cv[1])[:nlargest]
    output_matrix = principalDf.apply(sort_decr2_topn, axis=1)

    # write to file
    if os.path.exists(gestures_folder + "/task1.csv"):
        os.remove(gestures_folder + "/task1.csv")
    for i in range(0, len(output_matrix)):
        tmp_score_list = []
        for j in range(0, len(output_matrix[i])):
            tmp_str = str(output_matrix[i][j]).replace("'", "")
            tmp_score_list.append(tmp_str)
        with open(gestures_folder + "/task1.csv", 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(tmp_score_list)


def task1(gesture_folder, k, n, m, c=0.8):
    """
    :param gesture_folder: gesture_folder path
    :param k: top k similarity gestures
    :param n: user-defined seed nodes
    :param m: most dominant gestures
    :param c: probability value (predefined)
    :return: a dictionary after applying PPR algorithm and m most dominant gestures
    """
    # get a dictionary after applying PPR algorithm
    dic = {}
    result_list = []
    file_path = gesture_folder + "/" + "task1.csv"

    with open(file_path) as file:
        lines = file.readlines()
        length = len(lines)
        for i in range(length):
            temp_list = re.split(r',\s*(?![^()]*\))', lines[i])
            alist = []
            for j in range(k):
                temp_str = temp_list[j][2:-2]
                alist.append(temp_str.split(",")[0])
            dic[alist[0]] = alist
            result_list.append(alist)

        dic = dict(collections.OrderedDict(sorted(dic.items())))
    # call the dictionary
    keys = []
    for key, value in dic.items():
        keys.append(key)

    A = []
    for key, value in dic.items():
        # diff_list = (list(list(set(keys)-set(value)) + list(set(keys)-set(value))))
        list_of_zeros = [0] * length
        for item in value:
            list_of_zeros[keys.index(item)] = 1 / k
        A.append(list_of_zeros)

    A = np.asarray(A).T  # transpose the matrix because the vectors should be in each column
    # print(A)
    list_of_random_items = []
    for i in range(n):
        preference_file = input("Please enter preference gesture for " + str(i) + " (without suffix):\t")
        list_of_random_items.append(preference_file)

    # set the start vector
    vq = [0] * length
    for item in list_of_random_items:
        vq[keys.index(item)] = 1 / n

    vq = np.asarray(vq)
    uq = vq
    while True:
        temp = (1 - c) * np.dot(A, uq) + c * vq
        # if sum(abs(temp-uq)) < (0.0000001 * length):
        #     uq = temp
        #     break
        # check for converge
        checker = True
        for i in range(length):
            if abs(temp[i] - uq[i]) > 0.0000001:
                checker = False
        if checker:
            break
        uq = temp
    # print(uq)
    temp_dict = {}
    for i in range(length):
        temp_dict[keys[i]] = uq[i]

    # pick m most dominant gestures
    m_dominant_gestures = []
    # print(temp_dict)
    sorted_dict = sorted(temp_dict, key=temp_dict.get, reverse=True)
    # print(sorted_dict)
    for r in sorted_dict[:m]:
        # print(r, temp_dict[r])
        m_dominant_gestures.append(r)
    # print(m_dominant_gestures)
    plot_m_dominant_gestures(gesture_folder, m_dominant_gestures)

    return dic, m_dominant_gestures


def plot_m_dominant_gestures(gesture_folder, m_dominant_gestures):
    """
    :param gesture_folder:
    :param m_dominant_gestures:
    :return:
    """
    # write graphs to file (empty the directory each time this task reruns)
    directory = gesture_folder + "/output_graphs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_list = [f for f in os.listdir(directory)]
    for f in file_list:
        os.remove(os.path.join(directory, f))

    # plot all graphs from each of the four directions
    components = ["W", "X", "Y", "Z"]
    for item in m_dominant_gestures:
        for comp in components:
            string = gesture_folder + "/" + comp
            files = os.listdir(string)
            for file in files:
                if file == str(item) + ".csv":
                    with open(string + "/" + file) as f:
                        lines = f.readlines()
                        alist = []
                        length = 0
                        for i in range(20):
                            line = lines[i].split(",")
                            list_of_floats = [float(item) for item in line]
                            alist.append(list_of_floats)
                            length = len(list_of_floats)

                        x = np.asarray([*range(1, length + 1, 1)])
                        plt.xlabel('Time Series')
                        plt.ylabel('Values')
                        plt.title("Graph of " + comp + "/" + file)

                        ax = plt.gca()
                        # Move bottom x-axis to centre, passing through (0,0)
                        ax.spines['bottom'].set_position('zero')
                        # Eliminate upper and right axes
                        ax.spines['right'].set_color('none')
                        ax.spines['top'].set_color('none')

                        for i in range(20):
                            plt.plot(x, alist[i], label="Sensor" + str(i + 1))

                        plt.legend(loc=5, prop={'size': 6})  # modify the legend's location and size
                        plt.xticks(
                            [x for x in range(1, length + 6) if x % 1 == 0])  # show every possible time on x axis
                        plt.savefig("./" + gesture_folder + "/output_graphs/" + comp + file + ".png")
                        # plt.show()


def task2(user_option, training_labels_file, vector_model, gesture_folder, ppr_dic, c, m, distance_matrix, list_files, raw_data):
    """
        Classification and label all other gesture
    Parameter:
    -----------------------------------------
    :param user_option:     1: knn, 2: PPR, 3: decision tree
    :param training_labels_file:     A training labels files that contain a set of labeled gestures
    :param vector_model:    1: tf, 2: tf_idf
    :param gesture_folder:  Database contains all gestures
    :param ppr_dic:         ppr algorithm computed in task1
    :param c:               beta value
    :param m:               m-dominant
    :param distance_matrix:     distance_matrix
    :return:        all labeled gestures

    Method:
    -----------------------------------------
    PPR(personalize PageRank):

    KNN(k-nearest neighbor):
        Utilize distance comparision, distance function is standard Euclidean distance:
                    dis(A,B) = sqrt(((Ai - Bi) - (Ai - Bi)) for every i in vector)

    Decision Tree:

    """
    if user_option != "1" and user_option != "2" and user_option != "3":
        print("Choose the right option")
        return
    else:
        labels_dic = {}
        # read training data
        training_data = pd.read_excel(training_labels_file, header=None).values.tolist()
        # parser and store in dictionary
        for file_name, class_name in training_data:
            labels_dic[str(file_name)] = class_name

        # take only filename
        list_files = [filename.split(".")[0].split("-")[2] for filename in list_files]
        num_files = len(list_files)

        # knn
        if user_option == "1":
            k = int(input("Please enter a k:\t"))
            for first_vector_index, first_vector in enumerate(raw_data):
                if list_files[first_vector_index] not in labels_dic:
                    print("Associate a label for " + list_files[first_vector_index])
                    temp_distance_dic = {}
                    for second_vector_index, second_vector in enumerate(raw_data):
                        # distance = 0
                        # for i in range(len(first_vector)):
                        #     distance += math.pow((first_vector[i] - second_vector[i]), 2)
                        temp_distance_dic[list_files[second_vector_index]] = distance_matrix[first_vector_index][second_vector_index]
                    check_list = []
                    for file, value in temp_distance_dic.items():
                        check_list.append((file, value))
                    check_list = sorted(check_list, key=lambda x: x[1])
                    temp_dic = {}
                    temp_k = 0
                    for file, _ in check_list:
                        if file in labels_dic:
                            if labels_dic[file] not in temp_dic:
                                temp_dic[labels_dic[file]] = 1
                            else:
                                temp_dic[labels_dic[file]] += 1
                            temp_k += 1
                        if k == temp_k:
                            break
                    labels_dic[list_files[first_vector_index]] = sorted(temp_dic.items(),
                                                                        key=lambda x: x[1], reverse=True)[0][0]

        # PPR
        if user_option == "2":
            k = len(list(ppr_dic.items())[0][1])
            # print(k)
            transition_matrix = []
            keys = list(ppr_dic.keys())
            # print(keys)
            for vertex, targets in ppr_dic.items():
                list_of_zeros = [0] * num_files
                for target in targets:
                    list_of_zeros[keys.index(int(target))] = 1 / k
                transition_matrix.append(list_of_zeros)
            transition_matrix = np.asarray(transition_matrix).T

            for gesture_file in list_files:
                if gesture_file not in labels_dic:
                    vq = [0] * num_files
                    vq[keys.index(int(gesture_file))] = 1
                    vq = np.asarray(vq)
                    uq = vq
                    while True:
                        temp = (1 - c) * np.dot(transition_matrix, uq) + c * vq
                        checker = True
                        for i in range(num_files):
                            if abs(temp[i] - uq[i]) > 0.0000001:
                                checker = False
                        if checker:
                            break
                        uq = temp
                    temp_dict = {}
                    for i in range(num_files):
                        temp_dict[keys[i]] = uq[i]

                    # pick m most dominant gestures
                    sorted_dict = sorted(temp_dict, key=temp_dict.get, reverse=True)
                    temp_dic = {}
                    temp_m = 0
                    for file in sorted_dict:
                        if str(file) in labels_dic:
                            if labels_dic[str(file)] not in temp_dic:
                                temp_dic[labels_dic[str(file)]] = 1
                            else:
                                temp_dic[labels_dic[str(file)]] += 1
                            temp_m += 1
                        if m == temp_m:
                            break
                    # print(gesture_file)
                    # print(temp_dic)
                    labels_dic[gesture_file] = sorted(temp_dic.items(),
                                                      key=lambda x: x[1], reverse=True)[0][0]

        # decision tree
        if user_option == "3":
            _, list_files, raw_data = dataMatrix(gesture_folder + "/output_vectors", vector_model)
            list_files = [filename.split(".")[0].split("-")[2] for filename in list_files]
            labels_dic = phase3_decisiontree.decision_tree_train_test(labels_dic, list_files, raw_data)
        # print(labels_dic)

    # testing for accuracy:
    count = 0
    outputString = ""
    for key, value in labels_dic.items():
        temp = int(key.split("_")[0])
        if (temp <= 100 and value == "vattene") or (temp <= 400 and value
                                                    == "combinato") or (temp <= 700 and value == "daccordo"):
            count += 1
        outputString += key + " : " + value + "\n"
    print("Accuracy:")
    print(count / num_files)
    with open(gestures_folder + "/task2.txt", 'w') as myfile:
        myfile.write(outputString)
    return labels_dic


def euclidean_distance(raw_data, gestures_folder):
    """
            Compute Euclidean distance between each object
                    L2 = ||Oj(i) - Ok(i)|| for each i in vector
                    Norm-2 function
    :param raw_data:
    :return:        Euclidean distance matrix
    """
    num_files = len(raw_data)
    distance_dic = {}
    euclidean_distance_matrix = np.zeros((num_files, num_files), dtype="float64")
    for first_file_index, first_vector in enumerate(raw_data):
        print(first_file_index)
        for second_file_index, second_vector in enumerate(raw_data):
            two_file_string1 = str(first_file_index) + "_" + str(second_file_index)
            two_file_string2 = str(second_file_index) + "_" + str(first_file_index)
            if two_file_string1 in distance_dic:
                euclidean_distance_matrix[first_file_index][second_file_index] = distance_dic[two_file_string1]
            elif two_file_string2 in distance_dic:
                euclidean_distance_matrix[first_file_index][second_file_index] = distance_dic[two_file_string2]
            else:
                distance = 0
                for i in range(len(first_vector)):
                    # distance += (first_vector[i] - second_vector[i]) ** 2
                    distance += math.pow((first_vector[i] - second_vector[i]), 2)
                distance = math.sqrt(distance)
                euclidean_distance_matrix[first_file_index][second_file_index] = distance
                distance_dic[two_file_string1] = distance
    # save Euclidean Distance Matrix to file with 8 decimal places saved
    np.savetxt(gestures_folder + "/all_euclidean_distance.csv", euclidean_distance_matrix, delimiter=",", fmt='%1.8f')
    np.savetxt(gestures_folder + "/raw_data.csv", raw_data, delimiter=",", fmt='%1.8f')


    return euclidean_distance_matrix


def hash_function_family_checker(similarity_matrix, raw_data, direction, list_files):
    """
            Check for hash_function_family, two condition:
                1. sim(A, B) >= 0.6, Pr(H(A) = H(B)) >= 0.7
                2. sim(A, B) <= 0.2, Pr(H(A) = H(B)) <= 0.3

    :param similarity_matrix:       A given similarity_matrix using DTW         n*n
    :param raw_data:                Original dataset                    n*m
    :param direction:               hash direction                      1*m
    :param list_files:              list of gesture file in database
    :return:                    True if satisfy the condition, False otherwise
    """
    projected_points = np.dot(raw_data, direction.T)
    # print(projected_points)
    num_partitions, bins, num_files = 2, [], len(list_files)
    partition = (max(projected_points) - min(projected_points)) / num_partitions
    for i in range(num_partitions + 1):
        bins.append(i * partition + min(projected_points))
    bins = np.asarray(bins)
    converted_bins = []
    for file_index in range(len(projected_points)):
        count = 0
        for check_index in range(1, num_partitions + 1):
            if bins[check_index - 1] <= projected_points[file_index] <= bins[check_index]:
                converted_bins.append(count)
                break
            if check_index == num_partitions:
                converted_bins.append(count)
                break
            count += 1
    match, miss = 0, 0
    for first_index in range(num_files):
        max_distance = max(similarity_matrix[first_index])
        for second_index in range(num_files):
            # if similarity_matrix[first_index][second_index] >= 0.6 and abs(converted_bins[first_index] - converted_bins[second_index]) < 4:
            #     match += 1
            if similarity_matrix[first_index][second_index] >= (4 * max_distance / 5) and converted_bins[first_index] == \
                    converted_bins[second_index]:
                miss += 1
            else:
                match += 1

    # print(match)
    # print(miss)
    if match / (match + miss) < 0.6 or miss / (miss + match) > 0.4:
        return False
    else:
        return True


def task3(gesture_folder, num_layer, num_hashes, vector_model, t, gesture_file, similarity_matrix, list_files, raw_data):
    """
            LSH algorithm and using LSH result for finding t similar gestures for given gesture
    :param gesture_folder:      A given gesture folder
    :param num_layer:           Number of layers
    :param num_hashes:          Number of hashes in each layers
    :param vector_model:        tf or tf_idf
    :param t:                   number of similar gestures being returned
    :param gesture_file:        A given gesture in query
    :param similarity_matrix    gesture-gesture similarity matrix
    :return:
    """
    list_words, _, _ = dataMatrix(gesture_folder + "/output_vectors", vector_model)
    list_files = [filename.split(".")[0].split("-")[2] for filename in list_files]
    hash_directions = []
    # hash_function_family_checker(similarity_matrix, raw_data, np.random.rand(len(list_words)), list_files)
    for i in range(num_layer * num_hashes):
        while True:
            temp = np.random.rand(len(list_words))
            if hash_function_family_checker(similarity_matrix, raw_data, temp, list_files):
                hash_directions.append(temp)
                break
    hash_directions = np.asarray(hash_directions)  # (num_layer * num_hashes) * m
    all_ranges = np.dot(raw_data, hash_directions.T).T  # (num_layer * num_hashes) * n
    hash_range = []
    num_directions = 2  # int(math.pow(2, num_hashes))
    for range_length in all_ranges:
        temp_range = []
        partition = (max(range_length) - min(range_length)) / num_directions
        for i in range(num_directions + 1):
            temp_range.append(i * partition + min(range_length))
        hash_range.append(temp_range)
    hash_range = np.asarray(hash_range)  # (num_layer * num_hashes) * 8
    dic = {}
    for file_index, file in enumerate(list_files):
        all_layers = []
        for layer_index in range(num_layer):
            layer = []
            for hash_index in range(num_hashes):
                projected_point = np.dot(raw_data[file_index], hash_directions[layer_index * num_hashes + hash_index])
                count = 0
                for check_index in range(1, num_directions + 1):
                    if hash_range[layer_index * num_hashes + hash_index][check_index - 1] <= projected_point <= \
                            hash_range[layer_index * num_hashes + hash_index][check_index]:
                        layer.append(bin(count))
                        break
                    if check_index == num_directions:
                        layer.append(bin(count))
                        break
                    count += 1
            all_layers.append(layer)
        dic[file] = all_layers
    # print(dic)
    temp_result = {}
    results = []
    for key, value in dic.items():
        query_value = dic[gesture_file]
        for layer_index in range(len(value)):
            if query_value[layer_index] == value[layer_index]:
                if key not in temp_result:
                    temp_result[key] = 1
                else:
                    temp_result[key] += 1
    while len(temp_result.keys()) < t:
        _, temp_result = task3(gesture_folder, num_layer, num_hashes, vector_model, t, gesture_file, similarity_matrix, list_files, raw_data)

    # print(temp_result)
    print("unique and overall number of gestures:\t" + str(len(temp_result.keys())))
    result = [key for key, v in sorted(temp_result.items(), key=lambda item: item[1], reverse=True)][:t]
    print(result)
    outputString = "Top " + str(t) + " similar gesture files are:\n"
    for g in result:
        outputString += g + "\n"

    with open(gestures_folder + "/task3.txt", 'w') as myfile:
        myfile.write(outputString)
    return temp_result.keys(), temp_result


def ppr(ppr_dic, list_relevant_gesture, list_irrelevant_gesture, option, c):
    num_files = len(ppr_dic.keys())
    k = len(list(ppr_dic.items())[0][1])
    # print(k)
    transition_matrix = []
    keys = list(ppr_dic.keys())
    # print(keys)
    for vertex, targets in ppr_dic.items():
        list_of_zeros = [0] * num_files
        for target in targets:
            list_of_zeros[keys.index(int(target))] = 1 / k
        transition_matrix.append(list_of_zeros)
    transition_matrix = np.asarray(transition_matrix).T

    vq = [0.5] * num_files
    if option == 1:
        for gesture_file in list_relevant_gesture:
            vq[keys.index(int(gesture_file))] = 1
        for gesture_file in list_irrelevant_gesture:
            vq[keys.index(int(gesture_file))] = 0
    else:
        for gesture_file in list_relevant_gesture:
            vq[keys.index(int(gesture_file))] = 0
        for gesture_file in list_irrelevant_gesture:
            vq[keys.index(int(gesture_file))] = 1
    vq = np.asarray(vq)
    # normalization:
    vq = vq / sum(vq)
    # print(sum(vq))
    uq = vq
    while True:
        temp = (1 - c) * np.dot(transition_matrix, uq) + c * vq
        checker = True
        for i in range(num_files):
            if abs(temp[i] - uq[i]) > 0.0000001:
                checker = False
        if checker:
            break
        uq = temp
    temp_dict = {}
    for i in range(num_files):
        temp_dict[keys[i]] = uq[i]
    return temp_dict


def task5(ppr_dic, task3_output, c=0.8):
    """

    :param ppr_dic:
    :param task3_output:
    :param c:
    :return:
    """
    print("Performing task5")
    list_relevant_gesture = input("Please enter relevant gesture file (without suffix and use \",\" to separate):\t")
    list_irrelevant_gesture = input(
        "Please enter irrelevant gesture file (without suffix and use \",\" to separate):\t")
    list_relevant_gesture = list_relevant_gesture.split(",")
    list_irrelevant_gesture = list_irrelevant_gesture.split(",")
    # check does it in task3 output
    for gesture in list_relevant_gesture:
        if gesture not in task3_output:
            print(gesture + " is not in task3 output, execute this task again")
            return
    for gesture in list_irrelevant_gesture:
        if gesture not in task3_output:
            print(gesture + " is not in task3 output, execute this task again")
            return
    relevant_dic = ppr(ppr_dic, list_relevant_gesture, list_irrelevant_gesture, 1, c)
    irrelevant_dic = ppr(ppr_dic, list_relevant_gesture, list_irrelevant_gesture, 2, c)
    result = []
    for k, v in relevant_dic.items():
        if v >= irrelevant_dic[k]:
            result.append((k, v))
    result = sorted(result, key=lambda x: x[1], reverse=True)
    print(result)
    print("Number of relevant gestures:\n" + str(len(result)))
    return result


def task4a(gestures_folder, target_sensor_id):
    """
    :param gestures_folder:     Given gestures folder
    :return:
    """
    components = ["X", "Y", "Z", "W"]
    list_word = []  # store all unique tuple
    dic = {}  # general dictionary
    idf_dic = {}
    total_number_gesture_file = 0

    subfolder = gestures_folder + "/X"
    allFiles = os.listdir(subfolder)
    for file in allFiles:
        if file.endswith(".wrd.csv"):
            total_number_gesture_file += 1
            fileDic = {}
            for cl in components:
                allData = pd.read_csv(gestures_folder + "/" + cl + "/" + file, header=None, sep='\n').values.tolist()
                # parser of data
                for index, row in enumerate(allData):
                    allData[index] = row[0].split(",")
                row_num = len(allData)
                col_num = len(allData[2])
                # print(allData)
                for row_index in range(2, row_num):
                    sensor_id = allData[row_index][0]
                    if sensor_id == str(target_sensor_id):
                        h = 4
                        while h < col_num:
                            temp_word = "(" + cl + ", " + sensor_id + ", " + allData[row_index][h] + ")"
                            # update dictionary
                            if temp_word not in list_word:
                                list_word.append(temp_word)
                            # update n for tf count
                            if temp_word not in fileDic:
                                fileDic[temp_word] = 1
                                # update n for idf count
                                if temp_word not in idf_dic:
                                    idf_dic[temp_word] = 1
                                else:
                                    idf_dic[temp_word] += 1
                            else:
                                fileDic[temp_word] += 1
                            h += 2
            dic[file] = fileDic

    output_folder = gestures_folder + "/output_vectors_" + str(target_sensor_id)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for file_name, value in dic.items():
        temp = file_name.split(".")
        word_String = ""
        tf_String = ""
        tf_idf_String = ""
        total_count = sum(value.values())
        for word in list_word:
            word_String += word + "|"
            if word in value:
                tf_String += str(value[word] / total_count) + ","
                tf_idf_String += str(
                    (value[word] / total_count) * math.log10(total_number_gesture_file / idf_dic[word])) + ","
            else:
                tf_String += str(0.0) + ","
                tf_idf_String += str(0.0) + ","
        word_String = word_String[:-1]  # format output
        tf_String = tf_String[:-1]
        tf_idf_String = tf_idf_String[:-1]
        word_String += "\n\n<"
        tf_String += ">\n"
        tf_idf_String += ">\n"
        tf_outputString = word_String + tf_String
        filename = output_folder + "/tf-vectors-" + temp[0] + ".txt"
        with open(filename, 'w') as output_file:
            output_file.write(tf_outputString)

        tf_idf_outputString = word_String + tf_idf_String
        filename = output_folder + "/tfidf-vectors-" + temp[0] + ".txt"
        with open(filename, 'w') as output_file:
            output_file.write(tf_idf_outputString)

    return  # done for task 4a


def task4b(gestures_folder, components):
    """
    :param gestures_folder:     Given gestures folder
    :return:
    """
    # components = ["X", "Y", "Z", "W"]
    list_word = []  # store all unique tuple
    dic = {}  # general dictionary
    idf_dic = {}
    total_number_gesture_file = 0

    subfolder = gestures_folder + "/X"
    allFiles = os.listdir(subfolder)
    for file in allFiles:
        if file.endswith(".wrd.csv"):
            total_number_gesture_file += 1
            fileDic = {}
            for cl in components:
                allData = pd.read_csv(gestures_folder + "/" + cl + "/" + file, header=None, sep='\n').values.tolist()
                # parser of data
                for index, row in enumerate(allData):
                    allData[index] = row[0].split(",")
                row_num = len(allData)
                col_num = len(allData[2])
                # print(allData)
                for row_index in range(2, row_num):
                    sensor_id = allData[row_index][0]
                    h = 4
                    while h < col_num:
                        temp_word = "(" + cl + ", " + sensor_id + ", " + allData[row_index][h] + ")"
                        # update dictionary
                        if temp_word not in list_word:
                            list_word.append(temp_word)
                        # update n for tf count
                        if temp_word not in fileDic:
                            fileDic[temp_word] = 1
                            # update n for idf count
                            if temp_word not in idf_dic:
                                idf_dic[temp_word] = 1
                            else:
                                idf_dic[temp_word] += 1
                        else:
                            fileDic[temp_word] += 1
                        h += 2
            dic[file] = fileDic

    output_folder = gestures_folder + "/output_vectors_" + components[0]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for file_name, value in dic.items():
        temp = file_name.split(".")
        word_String = ""
        tf_String = ""
        tf_idf_String = ""
        total_count = sum(value.values())
        for word in list_word:
            word_String += word + "|"
            if word in value:
                tf_String += str(value[word] / total_count) + ","
                tf_idf_String += str(
                    (value[word] / total_count) * math.log10(total_number_gesture_file / idf_dic[word])) + ","
            else:
                tf_String += str(0.0) + ","
                tf_idf_String += str(0.0) + ","
        word_String = word_String[:-1]  # format output
        tf_String = tf_String[:-1]
        tf_idf_String = tf_idf_String[:-1]
        word_String += "\n\n<"
        tf_String += ">\n"
        tf_idf_String += ">\n"
        tf_outputString = word_String + tf_String
        filename = output_folder + "/tf-vectors-" + temp[0] + ".txt"
        with open(filename, 'w') as output_file:
            output_file.write(tf_outputString)

        tf_idf_outputString = word_String + tf_idf_String
        filename = output_folder + "/tfidf-vectors-" + temp[0] + ".txt"
        with open(filename, 'w') as output_file:
            output_file.write(tf_idf_outputString)

    return  # done for task 4b


def task4(gestures_folder, vector_model, w, s, r, result_4, similar_boundary):
    query_id = input("Query gesture: \t")
    query_mode = input("Query mode (1) for sensor, and (2) for WXYZ:\t")
    target_matrix = [None]
    query_sub_id = ["0"]
    if query_mode == "1":
        query_sub_id[0] = input("Query sensor (1 to 20):\t")
        task4a(gestures_folder, int(query_sub_id[0]))
        list_words, list_files, raw_data = dataMatrix(gestures_folder + "/output_vectors_"+str(query_sub_id[0]), vector_model)
        if query_sub_id[0] not in result_4:
            result_4[query_sub_id[0]] = euclidean_distance(raw_data)

    elif query_mode == "2":
        query_sub_id[0] = input("Query Axis: (W), (X), (Y), (Z)")
        task4b(gestures_folder, [query_sub_id[0]])
        list_words, list_files, raw_data = dataMatrix(gestures_folder + "/output_vectors_"+query_sub_id[0], vector_model)
        if query_sub_id[0] not in result_4:
            result_4[query_sub_id[0]] = euclidean_distance(raw_data)

    else:
        return

    filenames = [filename_str.split('.')[0].split('-')[-1] for filename_str in list_files]
    for i in range(len(filenames)):
        if query_id == filenames[i]:
            query_id = i

    index_list = []
    for index in range(len(result_4[query_sub_id[0]])):
        index_list.append([index, result_4[query_sub_id[0]][query_id][index]])

    index_list.sort(key=lambda x: x[1])
    index_list = index_list[1: similar_boundary+2]

    query_file_name = filenames[query_id]

    index_list_files = []
    for i in range(len(index_list)):
        index_list_files.append(filenames[index_list[i][0]])

    index_list_name_cpy = []
    for i in range(len(index_list)):
        index_list_name_cpy.append([filenames[index_list[i][0]], index_list[i][1]])
    print(index_list_name_cpy)

    print("Performing feedback action:")
    for i in range(len(index_list)):
        rele = input("Is file {} relevant to file {}? [(1) for yes, (0) for no, (q) to quit]\t".format(index_list_files[i], query_file_name))
        if rele == '0':
            result_4[query_sub_id[0]][query_id][index_list[i][0]] = float('inf')
            result_4[query_sub_id[0]][index_list[i][0]][query_id] = float('inf')
        elif rele == 'q':
            break


if __name__ == '__main__':
    gestures_folder = input("Please enter the path of gestures folder:\t")
    w = int(input("Please enter a window length:\t"))
    s = int(input("Please enter shift length:\t"))
    r = int(input("Please enter a resolution:\t"))
    task0a(gestures_folder, w, s, r)
    task0b(gestures_folder)
    vector_model = input("Please enter a vector_model for future computation:\n\t1. TF\n\t2. TF_IDF\n")

    # print(list_files)
    # print(len(list_files))

    # euc_file_path = Path(gestures_folder + "/all_euclidean_distance.csv")
    # if not euc_file_path.exists():
    #     list_words, list_files, raw_data = dataMatrix(gestures_folder + "/output_vectors", vector_model)
    #     euclidean_distance(raw_data, gestures_folder)
    #     outputString = ""
    #     for file_name in list_files:
    #         outputString += file_name + ","
    #     outputString = outputString[:-1] + "\n"
    #     with open(gestures_folder + "/fileName.txt", "w") as myfile:
    #         myfile.write(outputString)
    # distance_matrix = []
    # with open(gestures_folder + "/all_euclidean_distance.csv", 'r') as euc_file:
    #     for row in euc_file.readlines():
    #         temp = row.rstrip('\n').split(',')
    #         temp_list = []
    #         for item in temp:
    #             temp_list.append(float(item))
    #         distance_matrix.append(temp_list)
    # distance_matrix = np.asarray(distance_matrix)
    #
    # raw_data = []
    # with open(gestures_folder + "/raw_data.csv", 'r') as euc_file:
    #     for row in euc_file.readlines():
    #         temp = row.rstrip('\n').split(',')
    #         temp_list = []
    #         for item in temp:
    #             temp_list.append(float(item))
    #         raw_data.append(temp_list)
    # raw_data = np.asarray(raw_data)
    #
    # list_files = []
    # with open(gestures_folder + "/fileName.txt", "r") as myfile:
    #     line = myfile.readline()
    #     for filename in line.split(","):
    #         list_files.append(filename)
    #
    #
    # print(distance_matrix)
    # print(len(list_files))
    # ppr_dic, m, task3_output, task3_dic = None, None, None, None
    result_4 = {}
    while True:
        option = input("Please choose following command:\n"
                       "\ttask  1:  1\n"
                       "\ttask  2:  2\n"
                       "\ttask  3:  3\n"
                       "\ttask  6:  6\n"
                       "\t   quit:  q\n"
                       "Your Option is: ")

        if option == "1":
            k = int(input("Please enter a value for k:\t"))
            n = int(input("Please enter number of seed nodes (value of n):\t"))
            m = int(input("Please enter number of most dominant gestures (value of m):\t"))
            write_to_file(gestures_folder, vector_model, k)
            ppr_dic, _ = task1(gestures_folder, k, n, m)
        if option == "2":
            user_option_for_task2 = input("Please enter one of classifier shown below:\n"
                                          "\t1: KNN\n"
                                          "\t2: PPR\n"
                                          "\t3: Decision Tree\n"
                                          "Your Option is: ")
            training_labels_file = input("Please enter path to file contains training labels for classification")
            # task2(user_option_for_task2, training_labels_file, vector_model, gestures_folder, ppr_dic, 0.8, m, distance_matrix, list_files, raw_data)
        if option == "3":
            L = int(input("Please enter number of layers (value of l):\t"))
            K = int(input("Please enter number of hash function in each layer (value of k):\t"))
            t = int(input("Please enter number of similar gesture file (value of t):\t"))
            gesture_file = input("Please enter gesture filename (without suffix):\t")
            # task3_output, task3_dic = task3(gestures_folder, L, K, vector_model, t, gesture_file, distance_matrix, list_files, raw_data)
        if option == "6":
            if task3_output is None:
                print("Execute task 3 first before execute task 4 and task 5")
            else:
                user_option_for_task6 = input("Please enter one of classifier shown below:\n"
                                              "\ttask4: 4\n"
                                              "\ttask5: 5\n"
                                              "Your Option is: ")
                if user_option_for_task6 == "4":
                    similar_boundary = int(input("Please enter a value for relavent boundary:\t"))
                    task4(gestures_folder, vector_model, w, s, r, result_4, int(similar_boundary))
                if user_option_for_task6 == "5":
                    temp_result = task5(ppr_dic, task3_output)
                while True:
                    user_feedback = input("Please provide feedback on query result (Positive or Negative): \n")
                    if user_feedback == "Positive":
                        print("Query result is satisfied.")
                        break
                    if user_feedback == "Negative":
                        user_option_for_revisiting = input("Please choose one of the queries to revisit shown below:\n"
                                                  "\ttask4: 4\n"
                                                  "\ttask5: 5\n"
                                                  "Your Option is: ")
                        task3_output = []
                        for gesture, _ in temp_result:
                            task3_output.append(gesture)
                        if user_option_for_revisiting == "4":
                            similar_boundary = int(input("Please enter a value for relavent boundary:\t"))
                            task4(gestures_folder, vector_model, w, s, r, result_4, int(similar_boundary))
                        if user_option_for_revisiting == "5":
                            temp_result = task5(ppr_dic, task3_output)

        if option == "q":
            print("Quitting the program\n\n")
            break


