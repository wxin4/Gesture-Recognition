import glob

import pandas as pd
import numpy as np
import math
import os
import csv
import gensim

from gensim.models import LsiModel, CoherenceModel
from scipy.integrate import quad
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import scipy
import matplotlib
import matlab
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
import lda
from gensim import corpora
from gensim import models
from gensim import similarities
import matplotlib.pyplot as plt
import pyLDAvis.gensim
import pickle
import pyLDAvis
import warnings
from copy import deepcopy
from IPython import embed
import time


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
        # print(numerator[0])
        # print(denominator[0])
        if i == 0:
            band[i] = -1 + 2 * (numerator[0] / denominator[0])
        else:
            band[i] = band[i - 1] + 2 * (numerator[0] / denominator[0])

    # print(band)
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
    # print(band_Length)
    # find the mid point of each band
    middle_point_band = np.zeros(len(band_Length), dtype='float64')
    for i in range(len(band_Length)):
        if i == 0:
            middle_point_band[i] = (-1.0 + band_Length[i]) / 2
        else:
            middle_point_band[i] = (band_Length[i - 1] + band_Length[i]) / 2
    # print(middle_point_band)
    # print("\n\n")
    components = ["X", "Y", "Z", "W"]
    for cl in components:
        subfolder = gestures_folder + "/" + cl
        # print(subfolder)
        allFiles = os.listdir(subfolder)
        # print(allFile)
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

                    # print(allData[sensor_id])
                    # print("\n\n")

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
                    # print(normalized_values)
                    # print("\n\n")
                    # print(quantized_values)
                    # print("\n\n")
                    std_i_j = math.sqrt(sum_SD / col_num)
                    output_row.append(str(sensor_id + 1))  # output sensor id      A
                    output_row.append(str(avg_i_j))  # output average amplitude      B
                    output_row.append(str(std_i_j))  # output standard deviations    C

                    t = 0
                    # avg_i_j_h = []
                    # win_i_j_h = []
                    while t < col_num and (t + w) <= col_num:
                        if sensor_id == 0:
                            output_list[1].append("avgQ_" + str(t))
                            output_list[1].append("winQ_" + str(t))
                        # output_row.append(
                        #     str(sum(normalized_values[t:t + w]) / w))
                        # output average quantized amplitude
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
                    # print(avg_i_j_h)
                    # print(win_i_j_h)
                    # print(output_row)
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

    # output:
    # print(total_number_gesture_file)
    # print(dic["1.wrd.csv"])
    # print(len(list_word))
    # print(len(idf_dic.keys()))
    # print(idf_dic["(X, 1, <333>)"])
    output_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for file_name, value in dic.items():
        temp = file_name.split(".")
        word_String = ""
        tf_String = ""
        tf_idf_String = ""
        total_count = sum(value.values())
        # print(str(total_count))
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
        filename = output_folder + "/tf_vectors_" + temp[0] + ".txt"
        with open(filename, 'w') as output_file:
            output_file.write(tf_outputString)

        tf_idf_outputString = word_String + tf_idf_String
        filename = output_folder + "/tfidf_vectors_" + temp[0] + ".txt"
        with open(filename, 'w') as output_file:
            output_file.write(tf_idf_outputString)

    print(len(list_word))
    # print(idf_dic)
    # print(total_number_gesture_file)
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
        filename_prefix += "tf_vectors_"
    else:
        filename_prefix += "tfidf_vectors_"
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
            # data = np.asarray(f.readline()[1:-2].split(","), dtype='float64')
            data = f.readline()[1:-2].split(",")
            allData.append(data)

    allData = np.asarray(allData, dtype='float64')

    return list_words, list_files, allData


def sortWord_Scores(list_words, latent_feature):
    """

    :param list_words:  list of words
    :param latent_feature:  list of scores corresonding to list of words
    :return:   an array with format: <words, score>
    """
    output_list = []
    temp_sorted = sorted(latent_feature, reverse=True)
    checked = np.zeros(len(temp_sorted), dtype='int')
    temp_index = []
    for v in temp_sorted:
        for index, value in enumerate(latent_feature):
            if v == value and checked[index] == 0:
                temp_index.append(index)
                checked[index] = 1
                break

    for index in temp_index:
        pair = "<" + str(list_words[index]) + "," + str(latent_feature[index]) + ">"
        output_list.append(pair)

    return output_list


def task1(gestures_folder, vector_model, k, user_option):
    """

    :param gestures_folder:  The path to a set of gesture files
    :param vector_model:    TF or TF_IDF: 1, 2
    :param k:      top k-dimension
    :param user_option:     1,2,3,4
    :return:    a matrix with top-k latent feature and <words, score> as column
    """
    print("Performing task 1")
    output_vector_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_vector_folder):
        print("execute task0b first")
        return
    list_words, _, raw_data = dataMatrix(output_vector_folder, vector_model)
    # PCA
    if user_option == "1":
        print("Performing PCA")
        if os.path.exists(gestures_folder + "/PCA_top_k_latent_topics.csv"):
            os.remove(gestures_folder + "/PCA_top_k_latent_topics.csv")
        pca = PCA()
        pca.fit_transform(raw_data)
        UT = -1 * pca.components_
        # print(max(UT[0]))
        # print(str(max(UT[0])))
        # for i in range(k):
        #     print("Sorting top-" + str(i) + " feature")
        #     output_list.append(sortWord_Scores(list_words, UT[i]))
        #     print("top-" + str(i) + " feature is sorted")
        principalDf = pd.DataFrame(UT[:k], columns=list_words)
        sort_decr2_topn = lambda row, nlargest=len(list_words): sorted(pd.Series(zip(principalDf.columns, row)),
                                                                       key=lambda cv: -cv[1])[:nlargest]
        output_matrix = principalDf.apply(sort_decr2_topn, axis=1)

        # write to file
        for i in range(0, len(output_matrix)):
            tmp_score_list = []
            for j in range(0, len(output_matrix[i])):
                tmp_score_list.append("(" + str(output_matrix[i][j][0]) + ", " + str(output_matrix[i][j][1]) + ")")
            with open(gestures_folder + "/PCA_top_k_latent_topics.csv", 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(tmp_score_list)

    # SVD
    if user_option == "2":
        print("Performing SVD")

        if os.path.exists(gestures_folder + "/SVD_top_k_latent_topics.csv"):
            os.remove(gestures_folder + "/SVD_top_k_latent_topics.csv")

        # SVD package
        u, sigma, vt = scipy.linalg.svd(raw_data)

        # transform into a [k * num_of_words] matrix
        # result = raw_data.T * u[:, : k] * sigmaK.I
        sigmaK = matlab.mat(np.eye(k) * sigma[:k])
        temp_matrix = np.dot(raw_data.T, u[:, : k])
        result = np.dot(temp_matrix, sigmaK.I)

        # rotate the matrix as [k * num_of_words] format
        result = np.rot90(np.fliplr(result))
        # convert the matrix into list for later use
        result_list = result.tolist()

        # create a list to hold the top-k latent semantics
        latent_semantics = []
        for i in range(k):
            latent_semantic = []
            for j in range(len(list_words)):
                # <word, score> format
                temp_String = "("
                temp_String += str(list_words[j]) + ", "
                temp_String += str(result_list[i][j])
                temp_String += ")"
                latent_semantic.append(temp_String)

            # sort by the score in descending order
            latent_semantic.sort(key=lambda x: float(x.split(",")[3][:-1]), reverse=True)
            latent_semantics.append(latent_semantic)

        # convert to matrix [k * num_of_words]
        output_matrix = np.asarray(latent_semantics)

        for i in range(0, len(output_matrix)):
            tmp_score_list = []
            for j in range(0, len(output_matrix[i])):
                tmp_score_list.append(output_matrix[i][j])
            with open(gestures_folder + "/SVD_top_k_latent_topics.csv", 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(tmp_score_list)

    # NMF
    if user_option == "3":
        print("Performing NMF")

        if os.path.exists(gestures_folder + "/NMF_top_k_latent_topics.csv"):
            os.remove(gestures_folder + "/NMF_top_k_latent_topics.csv")

        nmf = NMF(n_components=k, max_iter=100000, random_state=None)
        nmf.fit_transform(raw_data)
        H = nmf.components_  # R
        # print("H shape", H.shape)
        principalDf = pd.DataFrame(H[:k], columns=list_words)
        # print("principalDF shape", principalDf.shape)
        sort_decr2_topn = lambda row, nlargest=len(list_words): sorted(pd.Series(zip(principalDf.columns, row)),
                                                                       key=lambda cv: -cv[1])[:nlargest]
        output_list = principalDf.apply(sort_decr2_topn, axis=1)

        # print("output 0", len(output_matrix[0]))
        # print("len output matrix 0", len(output_list[0]))
        # print("output list", "\n", output_list)
        # print(output_list[0])
        # print(output_list[0][0])
        for i in range(0, len(output_list)):
            tmp_score_list = []
            for j in range(0, len(output_list[i])):
                tmp_str = str(output_list[i][j]).replace("'", "")
                tmp_score_list.append(tmp_str)
            with open(gestures_folder + "/NMF_top_k_latent_topics.csv", 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(tmp_score_list)

    # LDA
    if user_option == "4":
        print("Performing LDA")

        if os.path.exists(gestures_folder + "/LDA_top_k_latent_topics.csv"):
            os.remove(gestures_folder + "/LDA_top_k_latent_topics.csv")

        # extract all words from each gesture file
        word_lists = []  # store all words from each gesture file
        list_word = []  # store all unique tuple
        components = ["X", "Y", "Z", "W"]
        subfolder = gestures_folder + "/X"
        allFiles = os.listdir(subfolder)
        # sorted(allFiles, key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        for file in allFiles:
            if file.endswith(".wrd.csv"):
                temp_str = ""
                word_list = []
                for cl in components:
                    allData = pd.read_csv(gestures_folder + "/" + cl + "/" + file, header=None,
                                          sep='\n').values.tolist()
                    # parser of data
                    for index, row in enumerate(allData):
                        allData[index] = row[0].split(",")
                    row_num = len(allData)
                    col_num = len(allData[2])

                    for row_index in range(2, row_num):
                        sensor_id = allData[row_index][0]
                        h = 4
                        while h < col_num:
                            temp_word = "(" + cl + ", " + sensor_id + ", " + allData[row_index][h] + ")"
                            if temp_word not in list_word:
                                list_word.append(temp_word)
                            temp_str += "  " + temp_word
                            h += 2

                word_list.append(temp_str.lstrip())
                word_lists.append(word_list)

        # each gesture file has different words and different length of words
        texts = [[word for word in doc[0].split("  ")] for doc in word_lists]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]  # all words

        # build LDA model (sequence not random, top-k latent semantics)
        Lda = gensim.models.ldamodel.LdaModel

        # IMPORTANT!!! random_state should be 1 and min_prob should be 0 to include all words
        lda_model = Lda(corpus=corpus, num_topics=k, random_state=1, id2word=dictionary,
                        minimum_probability=0.0, passes=4, alpha=[0.01] * k,
                        eta=[0.01] * len(dictionary.keys()))

        lda_model.save("Task1_LDA_MODEL")
        corpora.MmCorpus.serialize("Task1_CORPUS", corpus)

        # get all output
        topic_prob_list = lda_model.print_topics(num_topics=k, num_words=len(list_word))

        # extract and form a <word, score> pair with a matrix of [k * total_num_of_words]
        word_score_list = []
        for i in range(k):
            temp_list = []
            for j in range(len(list_word)):
                each_score_word_str = topic_prob_list[i][1].split(" + ")[j]
                score = each_score_word_str.split("*")[0]
                word = each_score_word_str.split("*")[1][1:-1]
                temp_str = "(" + word + ", " + score + ")"
                temp_list.append(temp_str)
            word_score_list.append(temp_list)

        output_matrix = np.asarray(word_score_list)

        for i in range(0, len(output_matrix)):
            tmp_score_list = []
            for j in range(0, len(output_matrix[i])):
                tmp_score_list.append(output_matrix[i][j])
            with open(gestures_folder + "/LDA_top_k_latent_topics.csv", 'a', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(tmp_score_list)

        # EXTRA!!!!! Visualize the topics
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(LDAvis_prepared, 'LDA_Visualization.html')
    # print(output_matrix)
    # return output_matrix


def DTW_Score(time_series_1, time_series_2):
    """

    :param time_series_1:   The first time series   with size n
    :param time_series_2:   The second time series   with size m
    :return:   bottom right score
    """
    DTW_matrix = np.zeros((len(time_series_1) + 1, len(time_series_2) + 1), dtype='float64')
    # print(time_series_1)
    # print(time_series_2)
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
    # print(DTW_matrix)
    return DTW_matrix[len(time_series_1)][len(time_series_2)]


def ED_CORE(str1, str2, i, j, r):
    """

    :param str1:   first string
    :param str2:   second string
    :param i:      index of str1
    :param j:      index of str2
    :return:
    """
    if i == -1 and j == -1:
        return 0
    elif i == -1 or j == -1:
        return 100000
    elif str1[i] == str2[j]:
        return ED_CORE(str1, str2, i - 1, j - 1, r)
    else:
        deleteCost = int(r / 2) + ED_CORE(str1, str2, i - 1, j, r)
        addCost = int(r / 2) + ED_CORE(str1, str2, i, j - 1, r)
        replaceCost = abs(int(str1[i]) - int(str2[j])) + ED_CORE(str1, str2, i - 1, j - 1, r)
        return min(deleteCost, addCost, replaceCost)


def task2(gestures_folder, gesture_file, vector_model, user_option, k, r):
    """

    :param gestures_folder: Path to databse
    :param gesture_file:    Given gesture file
    :param vector_model:    TF or TF_IDF: 1, 2
    :param user_option:     1,2,3,4,5,6,7
    :param k:      top k-dimension
    :param r:      resolution
    :return:
    """
    print("Performing task 2")
    output_vector_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_vector_folder):
        print("execute task0b first")
        return
    list_words, list_files, raw_data = dataMatrix(output_vector_folder, vector_model)
    filename = ""
    if vector_model == "1":
        filename += "tf_vectors_" + gesture_file + ".txt"
        model = "tf"
    else:
        filename += "tfidf_vectors_" + gesture_file + ".txt"
        model = "tfidf"
    if filename not in list_files:
        print("No such file is found")
        return
    file_index = 0          # used to store the index of query gesture file
    for index, file in enumerate(list_files):
        if file == filename:
            file_index = index
            break
    # print(len(list_files))
    # print(file_index)
    similarity = {}
    output = []
    # print(file_index)
    # Dot product
    if user_option == "1":
        print("Performing Dot product")
        for index, file in enumerate(list_files):
            similarity[file] = np.dot(raw_data[file_index], raw_data[index])

        ##### sort similarity
        for file, value in similarity.items():
            output.append((file.split(".")[0].split("_")[2], value))

    # PCA
    if user_option == "2":
        task1_output = gestures_folder + "/PCA_top_k_latent_topics.csv"
        if not os.path.exists(task1_output):
            print("execute task1 first")
            return
        latent_semantic_list = []
        df = pd.read_csv(gestures_folder + "/PCA_top_k_latent_topics.csv", header=None)
        for i in range(0, len(df)):
            wrd_score_dic = {}
            for j in range(0, len(df.columns)):
                tmp = df.iloc[i, j]
                tmp = tmp.split("), ")
                tmp_wrd = "(" + str(tmp[0]).replace("(", "") + ")"
                tmp_score = str(tmp[1]).replace(")", "")
                wrd_score_dic[tmp_wrd] = tmp_score
            # print(wrd_score_dic)
            latent_semantic_list.append(wrd_score_dic)
        # print(latent_semantic_list)
        latent_feature_list = []
        for i in range(len(latent_semantic_list)):
            latent_feature = []
            for word in list_words:
                latent_feature.append(float(latent_semantic_list[i][word]))
            latent_feature_list.append(latent_feature)
        latent_feature_list = np.asarray(latent_feature_list)

        new_projected_data = np.asarray(np.asmatrix(raw_data) * np.asmatrix(np.transpose(latent_feature_list)))

        for index, file in enumerate(list_files):
            # similarity[file] = np.dot(new_projected_data[file_index], new_projected_data[index])
            numerator = 0
            denominator = 0
            for i in range(len(new_projected_data[file_index])):
                numerator += min(abs(new_projected_data[file_index][i]), abs(new_projected_data[index][i]))
                denominator += max(abs(new_projected_data[file_index][i]), abs(new_projected_data[index][i]))
            similarity[file] = numerator / denominator

        ##### sort similarity
        for file, value in similarity.items():
            output.append((file.split(".")[0].split("_")[2], value))

    # SVD
    if user_option == "3":
        # SVD package
        task1_output = gestures_folder + "/SVD_top_k_latent_topics.csv"
        if not os.path.exists(task1_output):
            print("execute task1 first")
            return
        latent_semantic_list = []
        df = pd.read_csv(gestures_folder + "/SVD_top_k_latent_topics.csv", header=None)
        for i in range(0, len(df)):
            wrd_score_dic = {}
            for j in range(0, len(df.columns)):
                tmp = df.iloc[i, j]
                tmp = tmp.split("), ")
                tmp_wrd = "(" + str(tmp[0]).replace("(", "") + ")"
                tmp_score = str(tmp[1]).replace(")", "")
                wrd_score_dic[tmp_wrd] = tmp_score
            # print(wrd_score_dic)
            latent_semantic_list.append(wrd_score_dic)
        # print(latent_semantic_list)
        latent_feature_list = []
        for i in range(len(latent_semantic_list)):
            latent_feature = []
            for word in list_words:
                latent_feature.append(float(latent_semantic_list[i][word]))
            latent_feature_list.append(latent_feature)
        latent_feature_list = np.asarray(latent_feature_list)

        new_projected_data = np.asarray(np.asmatrix(raw_data) * np.asmatrix(np.transpose(latent_feature_list)))

        for index, file in enumerate(list_files):
            # similarity[file] = np.dot(new_projected_data[file_index], new_projected_data[index])
            numerator = 0
            denominator = 0
            for i in range(len(new_projected_data[file_index])):
                numerator += min(abs(new_projected_data[file_index][i]), abs(new_projected_data[index][i]))
                denominator += max(abs(new_projected_data[file_index][i]), abs(new_projected_data[index][i]))
            similarity[file] = numerator / denominator

        ##### sort similarity
        for file, value in similarity.items():
            output.append((file.split(".")[0].split("_")[2], value))
    if user_option == "4":
        task1_output = gestures_folder + "/NMF_top_k_latent_topics.csv"
        if not os.path.exists(task1_output):
            print("execute task1 first")
            return
        latent_semantic_list = []
        df = pd.read_csv(gestures_folder + "/NMF_top_k_latent_topics.csv", header=None)
        for i in range(0, len(df)):
            wrd_score_dic = {}
            for j in range(0, len(df.columns)):
                tmp = df.iloc[i, j]
                tmp = tmp.split("), ")
                tmp_wrd = "(" + str(tmp[0]).replace("(", "") + ")"
                tmp_score = str(tmp[1]).replace(")", "")
                wrd_score_dic[tmp_wrd] = tmp_score
            # print(wrd_score_dic)
            latent_semantic_list.append(wrd_score_dic)
        # print(latent_semantic_list)
        latent_feature_list = []
        for i in range(len(latent_semantic_list)):
            latent_feature = []
            for word in list_words:
                latent_feature.append(float(latent_semantic_list[i][word]))
            latent_feature_list.append(latent_feature)
        latent_feature_list = np.asarray(latent_feature_list)

        new_projected_data = np.asarray(np.asmatrix(raw_data) * np.asmatrix(np.transpose(latent_feature_list)))

        for index, file in enumerate(list_files):
            # similarity[file] = np.dot(new_projected_data[file_index], new_projected_data[index])
            numerator = 0
            denominator = 0
            for i in range(len(new_projected_data[file_index])):
                numerator += min(abs(new_projected_data[file_index][i]), abs(new_projected_data[index][i]))
                denominator += max(abs(new_projected_data[file_index][i]), abs(new_projected_data[index][i]))
            similarity[file] = numerator / denominator

        ##### sort similarity
        for file, value in similarity.items():
            output.append((file.split(".")[0].split("_")[2], value))
    if user_option == "5":
        lda_model = models.LdaModel.load('TASK1_LDA_MODEL')
        corpus = corpora.MmCorpus("Task1_CORPUS")

        doc_topic_matrix = []
        for doc_index in range(len(list_files)):
            doc_topic_list = [prob[1] for prob in lda_model[corpus[doc_index]]]
            doc_topic_matrix.append(doc_topic_list)
            print(lda_model[corpus[doc_index]])
        doc_topic_matrix = np.asarray(doc_topic_matrix)
        # print(doc_topic_matrix)

        for index, file in enumerate(list_files):
            similarity[file] = np.dot(doc_topic_matrix[file_index], doc_topic_matrix[index])

        ##### sort similarity
        for file, value in similarity.items():
            output.append((file.split(".")[0].split("_")[2], value))
    # ED
    if user_option == "6":
        print("Performing Edit-Distance")
        # ED_cost = {   "add": int(r/2),
        #               "del": int(r/2)}
        #               "rep": abs(int(str1[i]) - int(str2[i]))

        # read winQ from gesture file in query
        components = ["X", "Y", "Z", "W"]
        original_raw_data = []
        for component in components:
            allData = pd.read_csv(gestures_folder + "/" + component + "/" + gesture_file + ".wrd.csv", header=None,
                                  sep='\n').values.tolist()
            # parser of data
            for index, row in enumerate(allData):
                allData[index] = row[0].split(",")
            row_num = len(allData)
            col_num = len(allData[2])
            num_col = int((col_num - 3) / 2)
            string_array = []  # used to store all sensor_string in component
            for row_index in range(2, row_num):
                sensor_string = []
                for col_index in range(num_col):
                    h = col_index * 2 + 4
                    sensor_string.append(allData[row_index][h][1:-1])
                string_array.append(sensor_string)
            original_raw_data.append(string_array)
        # print(original_raw_data)

        # read all files
        for file in list_files:
            filename = file.split(".")[0].split("_")[2]
            # print(filename)
            total_cost = 0
            for index, component in enumerate(components):
                allData = pd.read_csv(gestures_folder + "/" + component + "/" + filename + ".wrd.csv", header=None,
                                      sep='\n').values.tolist()
                # parser of data
                for i, row in enumerate(allData):
                    allData[i] = row[0].split(",")
                row_num = len(allData)
                col_num = len(allData[2])
                num_col = int((col_num - 3) / 2)
                for row_index in range(2, row_num):
                    tempString = []
                    for col_index in range(num_col):
                        h = col_index * 2 + 4
                        tempString.append(allData[row_index][h][1:-1])
                    sourceString = original_raw_data[index][row_index - 2]
                    longer_length = max(len(sourceString), len(tempString))
                    for word_length in range(longer_length):
                        if word_length < len(sourceString) and word_length < len(tempString):
                            total_cost += ED_CORE(tempString[word_length], sourceString[word_length],
                                                  len(tempString[word_length]) - 1, len(sourceString[word_length]) - 1,
                                                  r)
                        elif len(sourceString) <= word_length < len(tempString):
                            str_empty = "0" + tempString[word_length]
                            total_cost += ED_CORE(str_empty, "0", len(tempString[word_length]), 0, r)
                        elif len(tempString) <= word_length < len(sourceString):
                            str_empty = "0" + sourceString[word_length]
                            total_cost += ED_CORE("0", str_empty, 0, len(sourceString[word_length]), r)
                        else:
                            print("123123123")
                    # total_cost += ED_CORE(tempString, sourceString, len(tempString)-1, len(sourceString)-1, r)
                    if total_cost > 100000:
                        print("321321321")
                    #         print(total_cost)
                    #         print(sourceString)
                    #         print(tempString)
                    #         print(word_length)
                    #         return
            similarity[file] = total_cost
        ##### sort similarity
        maxV = max(similarity.values())
        minV = min(similarity.values())
        for file, value in similarity.items():
            temp = 1 - ((value - minV) / (maxV - minV))
            output.append((file.split(".")[0].split("_")[2], temp))

    # DTW
    if user_option == "7":
        print("Performing DTW")
        # literate each file
        components = ["X", "Y", "Z", "W"]
        original_raw_data = []  # used to store value for query file of different components
        for component in components:
            allData = pd.read_csv(gestures_folder + "/" + component + "/" + gesture_file + ".wrd.csv", header=None,
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
            # print(avg_component_data)
            original_raw_data.append(avg_component_data)

        for file in list_files:
            filename = file.split(".")[0].split("_")[2]
            score = 0
            for index, component in enumerate(components):
                allData = pd.read_csv(gestures_folder + "/" + component + "/" + filename + ".wrd.csv", header=None,
                                      sep='\n').values.tolist()
                # parser of data
                for i, row in enumerate(allData):
                    allData[i] = row[0].split(",")
                row_num = len(allData)
                col_num = len(allData[2])
                num_col = int((col_num - 3) / 2)
                for row_index in range(2, row_num):
                    avg_component_compare_data = np.zeros(num_col, dtype='float64')
                    for col_index in range(num_col):
                        h = col_index * 2 + 3
                        avg_component_compare_data[col_index] = float(allData[row_index][h])
                    # DTW algorithm:
                    temp = original_raw_data[index][row_index - 2]
                    score += DTW_Score(temp, avg_component_compare_data)
            similarity[file] = score

        ##### sort similarity
        maxV = max(similarity.values())
        minV = min(similarity.values())
        for file, value in similarity.items():
            temp = 1 - ((value - minV) / (maxV - minV))
            output.append((file.split(".")[0].split("_")[2], temp))
    output = sorted(output, key=lambda x: x[1], reverse=True)[0:10]

    outputString = ""
    for i in output:
        outputString += str(i) + "\n"
    with open(gestures_folder + "/task2_(" + gesture_file + "_" + user_option + ")_" + model + ".txt", "w") as outputfile:
        outputfile.write(outputString)


def convert_distance_to_similarity(distance_matrix):
    """

    :param distance_matrix:     Given distance_matrix
    :return:    a similarity matrix using normalization method
    """
    similarity_matrix = []
    # print(len(distance_matrix))
    # print(len(distance_matrix[0]))
    for i in range(len(distance_matrix)):
        normalized_value = np.zeros(len(distance_matrix[i]), dtype="float64")
        max_num = max(distance_matrix[i])
        min_num = min(distance_matrix[i])
        for y in range(len(distance_matrix[i])):
            normalized_value[y] = 1 - ((distance_matrix[i][y] - min_num) / (max_num - min_num))
        similarity_matrix.append(normalized_value)
    return np.asarray(similarity_matrix)


def generate_gesture_matrix(user_option, raw_data, list_files, gestures_folder, r, list_words):
    """

    :param user_option:     1,2,3,4,5,6,7
    :param user_option:     gesture-feature matrix
    :param list_files:      list of all files in database
    :param gestures_folder:  Given gestuers database
    :return:
    """
    list_files = [filename.split(".")[0].split("_")[2] for filename in list_files]
    # Dot product
    if user_option == "1":
        gestures_similarity_matrix = np.asarray(np.asmatrix(raw_data) * np.asmatrix(np.transpose(raw_data)))
    # PCA
    if user_option == "2":
        task1_output = gestures_folder + "/PCA_top_k_latent_topics.csv"
        if not os.path.exists(task1_output):
            print("execute task1 first")
            return
        latent_semantic_list = []
        df = pd.read_csv(gestures_folder + "/PCA_top_k_latent_topics.csv", header=None)
        for i in range(0, len(df)):
            wrd_score_dic = {}
            for j in range(0, len(df.columns)):
                tmp = df.iloc[i, j]
                tmp = tmp.split("), ")
                tmp_wrd = "(" + str(tmp[0]).replace("(", "") + ")"
                tmp_score = str(tmp[1]).replace(")", "")
                wrd_score_dic[tmp_wrd] = tmp_score
            # print(wrd_score_dic)
            latent_semantic_list.append(wrd_score_dic)
        # print(latent_semantic_list)
        latent_feature_list = []
        for i in range(len(latent_semantic_list)):
            latent_feature = []
            for word in list_words:
                latent_feature.append(float(latent_semantic_list[i][word]))
            latent_feature_list.append(latent_feature)
        latent_feature_list = np.asarray(latent_feature_list)

        new_projected_data = np.asarray(np.asmatrix(raw_data) * np.asmatrix(np.transpose(latent_feature_list)))
        num_files_in_database = len(list_files)
        gestures_similarity_matrix = np.zeros((num_files_in_database, num_files_in_database), dtype="float64")
        for index1, file1 in enumerate(list_files):
            for index2, file2 in enumerate(list_files):
                numerator = 0
                denominator = 0
                for i in range(len(new_projected_data[0])):
                    numerator += min(abs(new_projected_data[index1][i]), abs(new_projected_data[index2][i]))
                    denominator += max(abs(new_projected_data[index1][i]), abs(new_projected_data[index2][i]))
                gestures_similarity_matrix[index1][index2] = numerator / denominator

    # SVD
    if user_option == "3":
        task1_output = gestures_folder + "/SVD_top_k_latent_topics.csv"
        if not os.path.exists(task1_output):
            print("execute task1 first")
            return
        latent_semantic_list = []
        df = pd.read_csv(gestures_folder + "/SVD_top_k_latent_topics.csv", header=None)
        for i in range(0, len(df)):
            wrd_score_dic = {}
            for j in range(0, len(df.columns)):
                tmp = df.iloc[i, j]
                tmp = tmp.split("), ")
                tmp_wrd = "(" + str(tmp[0]).replace("(", "") + ")"
                tmp_score = str(tmp[1]).replace(")", "")
                wrd_score_dic[tmp_wrd] = tmp_score
            # print(wrd_score_dic)
            latent_semantic_list.append(wrd_score_dic)
        # print(latent_semantic_list)
        latent_feature_list = []
        for i in range(len(latent_semantic_list)):
            latent_feature = []
            for word in list_words:
                latent_feature.append(float(latent_semantic_list[i][word]))
            latent_feature_list.append(latent_feature)
        latent_feature_list = np.asarray(latent_feature_list)

        new_projected_data = np.asarray(np.asmatrix(raw_data) * np.asmatrix(np.transpose(latent_feature_list)))
        num_files_in_database = len(list_files)
        gestures_similarity_matrix = np.zeros((num_files_in_database, num_files_in_database), dtype="float64")
        for index1, file1 in enumerate(list_files):
            for index2, file2 in enumerate(list_files):
                numerator = 0
                denominator = 0
                for i in range(len(new_projected_data[0])):
                    numerator += min(abs(new_projected_data[index1][i]), abs(new_projected_data[index2][i]))
                    denominator += max(abs(new_projected_data[index1][i]), abs(new_projected_data[index2][i]))
                gestures_similarity_matrix[index1][index2] = numerator / denominator
    # NMF
    if user_option == "4":
        task1_output = gestures_folder + "/NMF_top_k_latent_topics.csv"
        if not os.path.exists(task1_output):
            print("execute task1 first")
            return
        latent_semantic_list = []
        df = pd.read_csv(gestures_folder + "/NMF_top_k_latent_topics.csv", header=None)
        for i in range(0, len(df)):
            wrd_score_dic = {}
            for j in range(0, len(df.columns)):
                tmp = df.iloc[i, j]
                tmp = tmp.split("), ")
                tmp_wrd = "(" + str(tmp[0]).replace("(", "") + ")"
                tmp_score = str(tmp[1]).replace(")", "")
                wrd_score_dic[tmp_wrd] = tmp_score
            # print(wrd_score_dic)
            latent_semantic_list.append(wrd_score_dic)
        # print(latent_semantic_list)
        latent_feature_list = []
        for i in range(len(latent_semantic_list)):
            latent_feature = []
            for word in list_words:
                latent_feature.append(float(latent_semantic_list[i][word]))
            latent_feature_list.append(latent_feature)
        latent_feature_list = np.asarray(latent_feature_list)

        new_projected_data = np.asarray(np.asmatrix(raw_data) * np.asmatrix(np.transpose(latent_feature_list)))
        num_files_in_database = len(list_files)
        gestures_similarity_matrix = np.zeros((num_files_in_database, num_files_in_database), dtype="float64")
        for index1, file1 in enumerate(list_files):
            for index2, file2 in enumerate(list_files):
                numerator = 0
                denominator = 0
                for i in range(len(new_projected_data[0])):
                    numerator += min(abs(new_projected_data[index1][i]), abs(new_projected_data[index2][i]))
                    denominator += max(abs(new_projected_data[index1][i]), abs(new_projected_data[index2][i]))
                gestures_similarity_matrix[index1][index2] = numerator / denominator
    # LDA
    if user_option == "5":
        task1_output = gestures_folder + "/LDA_top_k_latent_topics.csv"
        if not os.path.exists(task1_output):
            print("execute task1 first")
            return
        latent_semantic_list = []
        df = pd.read_csv(gestures_folder + "/LDA_top_k_latent_topics.csv", header=None)
        for i in range(0, len(df)):
            wrd_score_dic = {}
            for j in range(0, len(df.columns)):
                tmp = df.iloc[i, j]
                tmp = tmp.split("), ")
                tmp_wrd = "(" + str(tmp[0]).replace("(", "") + ")"
                tmp_score = str(tmp[1]).replace(")", "")
                wrd_score_dic[tmp_wrd] = tmp_score
            # print(wrd_score_dic)
            latent_semantic_list.append(wrd_score_dic)
        # print(latent_semantic_list)
        latent_feature_list = []
        for i in range(len(latent_semantic_list)):
            latent_feature = []
            for word in list_words:
                latent_feature.append(float(latent_semantic_list[i][word]))
            latent_feature_list.append(latent_feature)
        latent_feature_list = np.asarray(latent_feature_list)

        new_projected_data = np.asarray(np.asmatrix(raw_data) * np.asmatrix(np.transpose(latent_feature_list)))
        num_files_in_database = len(list_files)
        gestures_similarity_matrix = np.zeros((num_files_in_database, num_files_in_database), dtype="float64")
        for index1, file1 in enumerate(list_files):
            for index2, file2 in enumerate(list_files):
                numerator = 0
                denominator = 0
                for i in range(len(new_projected_data[0])):
                    numerator += min(abs(new_projected_data[index1][i]), abs(new_projected_data[index2][i]))
                    denominator += max(abs(new_projected_data[index1][i]), abs(new_projected_data[index2][i]))
                gestures_similarity_matrix[index1][index2] = numerator / denominator
    # ED
    if user_option == "6":
        num_files_in_database = len(list_files)
        ED_distance_matrix = np.zeros((num_files_in_database, num_files_in_database), dtype="float64")
        components = ["X", "Y", "Z", "W"]
        file_data = {}  # utilization of memorization in order to decrease time complexity
        cost_dic = {}  # utilization of memorization in order to decrease time complexity
        for first_index, first_file in enumerate(list_files):
            # print(first_index)
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

                    num_col = int((col_num - 3) / 2)
                    string_array = []
                    for row_index in range(2, row_num):
                        sensor_string = []
                        for col_index in range(num_col):
                            h = col_index * 2 + 4
                            sensor_string.append(allData[row_index][h][1:-1])
                        string_array.append(sensor_string)
                    first_file_raw_data.append(string_array)
                file_data[first_file] = first_file_raw_data
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

                        num_col = int((col_num - 3) / 2)
                        string_array = []
                        for row_index in range(2, row_num):
                            sensor_string = []
                            for col_index in range(num_col):
                                h = col_index * 2 + 4
                                sensor_string.append(allData[row_index][h][1:-1])
                            string_array.append(sensor_string)
                        second_file_raw_data.append(string_array)
                    file_data[second_file] = second_file_raw_data

                # compute distance
                first_file_data = file_data[first_file]
                second_file_data = file_data[second_file]
                temp_compare_string1 = first_file + ", " + second_file
                temp_compare_string2 = second_file + ", " + first_file

                # utilized memorization
                if temp_compare_string1 not in cost_dic and temp_compare_string2 not in cost_dic:
                    total_cost = 0
                    for component_index in range(4):  # 4 components
                        for sensor_index in range(20):  # 20 sensors
                            sourceString = first_file_data[component_index][sensor_index]
                            tempString = second_file_data[component_index][sensor_index]
                            longer_length = max(len(sourceString), len(tempString))
                            for word_length in range(longer_length):
                                if word_length < len(sourceString) and word_length < len(tempString):
                                    total_cost += ED_CORE(tempString[word_length], sourceString[word_length],
                                                          len(tempString[word_length]) - 1,
                                                          len(sourceString[word_length]) - 1,
                                                          r)
                                elif len(sourceString) <= word_length < len(tempString):
                                    str_empty = "0" + tempString[word_length]
                                    total_cost += ED_CORE(str_empty, "0", len(tempString[word_length]), 0, r)
                                elif len(tempString) <= word_length < len(sourceString):
                                    str_empty = "0" + sourceString[word_length]
                                    total_cost += ED_CORE("0", str_empty, 0, len(sourceString[word_length]), r)
                                else:
                                    print("123123123")
                    cost_dic[temp_compare_string1] = total_cost
                    cost_dic[temp_compare_string2] = total_cost
                ED_distance_matrix[first_index][second_index] = cost_dic[temp_compare_string1]
        gestures_similarity_matrix = convert_distance_to_similarity(ED_distance_matrix)

    # DTW
    if user_option == "7":
        num_files_in_database = len(list_files)
        DTW_distance_matrix = np.zeros((num_files_in_database, num_files_in_database), dtype="float64")
        components = ["X", "Y", "Z", "W"]
        file_data = {}  # utilization of memorization in order to decrease time complexity
        scores_dic = {}  # utilization of memorization in order to decrease time complexity
        for first_index, first_file in enumerate(list_files):
            # print(first_index)
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
                    # print(avg_component_data)
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

        # print(DTW_distance_matrix)
        # convert to similarity matrix by using normalization of range 0-1
        gestures_similarity_matrix = convert_distance_to_similarity(DTW_distance_matrix)

    print(gestures_similarity_matrix)
    return gestures_similarity_matrix


def task3a(gestures_folder, vector_model, user_option, k, p, r):
    """

    :param gestures_folder:     Given gestuers database
    :param vector_model:        TF or TFIDF: 1  2
    :param user_option:         1,2,3,4,5,6,7
    :param k:                   top k-dimension
    :param p:                   top p-dimension
    :param r:                   resolution
    :return:
    """
    print("Performing task 3a")
    output_vector_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_vector_folder):
        print("execute task0b first")
        return
    list_words, list_files, raw_data = dataMatrix(output_vector_folder, vector_model)

    # sub problem 1:
    gestures_similarity_matrix = generate_gesture_matrix(user_option, raw_data, list_files, gestures_folder, r, list_words)

    # sub problem 2:
    u, sigma, vt = scipy.linalg.svd(gestures_similarity_matrix)

    # sub problem 3:
    list_files = [filename.split(".")[0].split("_")[2] for filename in list_files]
    principalDf = pd.DataFrame(vt[:p], columns=list_files)
    sort_decr2_topn = lambda row, nlargest=len(list_files): sorted(pd.Series(zip(principalDf.columns, row)),
                                                                   key=lambda cv: -cv[1])[:nlargest]
    output_matrix = principalDf.apply(sort_decr2_topn, axis=1)
    # print(output_matrix)
    # write to file
    if os.path.exists(gestures_folder + "/task3a.csv"):
        os.remove(gestures_folder + "/task3a.csv")
    for i in range(0, len(output_matrix)):
        tmp_score_list = []
        for j in range(0, len(output_matrix[i])):
            tmp_str = str(output_matrix[i][j]).replace("'", "")
            tmp_score_list.append(tmp_str)
        with open(gestures_folder + "/task3a.csv", 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(tmp_score_list)


def task3b(gestures_folder, vector_model, user_option, k, p, r):
    """

    :param gestures_folder:     Given gestuers database
    :param vector_model:        TF or TFIDF: 1  2
    :param user_option:         1,2,3,4,5,6,7
    :param k:                   top k-dimension
    :param p:                   top p-dimension
    :param r:                   resolution
    :return:
    """
    print("Performing task 3b")
    output_vector_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_vector_folder):
        print("execute task0b first")
        return
    list_words, list_files, raw_data = dataMatrix(output_vector_folder, vector_model)

    # sub problem 1:
    gestures_similarity_matrix = generate_gesture_matrix(user_option, raw_data, list_files, gestures_folder, r, list_words)

    # sub problem 2:
    nmf = NMF(n_components=p, max_iter=100000, random_state=None)
    nmf.fit_transform(gestures_similarity_matrix)
    H = nmf.components_  # R
    # print("H shape", H.shape)

    # sub problem 3:
    list_files = [filename.split(".")[0].split("_")[2] for filename in list_files]
    principalDf = pd.DataFrame(H[:p], columns=list_files)
    # print("principalDF shape", principalDf.shape)
    sort_decr2_topn = lambda row, nlargest=len(list_words): sorted(pd.Series(zip(principalDf.columns, row)),
                                                                   key=lambda cv: -cv[1])[:nlargest]
    output_matrix = principalDf.apply(sort_decr2_topn, axis=1)

    for i in range(0, len(output_matrix)):
        tmp_score_list = []
        for j in range(0, len(output_matrix[i])):
            tmp_str = str(output_matrix[i][j]).replace("'", "")
            tmp_score_list.append(tmp_str)
        with open(gestures_folder + "/task3b.csv", 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(tmp_score_list)


def task4a(gestures_folder):
    """

        :param gestures_folder:     Given gestures database
        :return:
        """

    print("Performing task 4a")

    if os.path.exists(gestures_folder + "/task4a.csv"):
        os.remove(gestures_folder + "/task4a.csv")

    task3a_output = gestures_folder + "/task3a.csv"
    if not os.path.exists(task3a_output):
        print("execute task3a first")
        return

    gesture_p_score_dic = {}

    df = pd.read_csv(gestures_folder + "/task3a.csv", header = None)
    num_latent_semantic = len(df)
    for i in range(0, len(df.columns)):
        for j in range(0, len(df)):
            p_score_dic = []
            tmp = df.iloc[j, i].replace("(", "")
            tmp = tmp.replace(")", "")
            tmp = tmp.split(", ")
            tmp_gesture = tmp[0]
            tmp_score = tmp[1]
            p_score_dic.append(j)
            p_score_dic.append(tmp_score)
            if tmp_gesture in gesture_p_score_dic:
                tmp_list = gesture_p_score_dic[tmp_gesture]
                tmp_list.append(p_score_dic)
                gesture_p_score_dic[tmp_gesture] = tmp_list
            else:
                gesture_p_score_dic[tmp_gesture] = [p_score_dic]
    print(gesture_p_score_dic)
    gesture_p_dic = {}
    for ges, p_score in gesture_p_score_dic.items():
        tmp_max = -float("inf")
        tmp_max_index = 0
        for i in range(0, len(p_score)):
            if float(p_score[i][1]) >= tmp_max:
                tmp_max = float(p_score[i][1])
                tmp_max_index = i
        gesture_p_dic[ges] = p_score[tmp_max_index][0]
    print(gesture_p_dic)
    same_p_clustering = {}
    for ges1, p1 in gesture_p_dic.items():
        same_p_list = []
        same_p_list.append(ges1)
        for ges2, p2 in gesture_p_dic.items():
            if p1 == p2 and ges1 != ges2:
                same_p_list.append(ges2)
        same_p_clustering[p1] = same_p_list
    print(same_p_clustering)

    for i in range(0, num_latent_semantic):
        if i not in same_p_clustering:
            same_p_clustering[i] = []

    for p, gesture in same_p_clustering.items():
        with open(gestures_folder + "/task4a.csv", 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow([p])
            wr.writerow(gesture)


def task4b(gestures_folder):
    """

        :param gestures_folder:     Given gestures database
        :return:
        """

    print("Performing task 4b")

    if os.path.exists(gestures_folder + "/task4b.csv"):
        os.remove(gestures_folder + "/task4b.csv")

    task3b_output = gestures_folder + "/task3b.csv"
    if not os.path.exists(task3b_output):
        print("execute task3b first")
        return

    gesture_p_score_dic = {}

    df = pd.read_csv(gestures_folder + "/task3b.csv", header = None)
    num_latent_semantic = len(df)
    for i in range(0, len(df.columns)):
        for j in range(0, len(df)):
            p_score_dic = []
            tmp = df.iloc[j, i].replace("(", "")
            tmp = tmp.replace(")", "")
            tmp = tmp.split(", ")
            tmp_gesture = tmp[0]
            tmp_score = tmp[1]
            p_score_dic.append(j)
            p_score_dic.append(tmp_score)
            if tmp_gesture in gesture_p_score_dic:
                tmp_list = gesture_p_score_dic[tmp_gesture]
                tmp_list.append(p_score_dic)
                gesture_p_score_dic[tmp_gesture] = tmp_list
            else:
                gesture_p_score_dic[tmp_gesture] = [p_score_dic]
    print(gesture_p_score_dic)
    gesture_p_dic = {}
    for ges, p_score in gesture_p_score_dic.items():
        tmp_max = -float("inf")
        tmp_max_index = 0
        for i in range(0, len(p_score)):
            if float(p_score[i][1]) >= tmp_max:
                tmp_max = float(p_score[i][1])
                tmp_max_index = i
        gesture_p_dic[ges] = p_score[tmp_max_index][0]
    print(gesture_p_dic)
    same_p_clustering = {}
    for ges1, p1 in gesture_p_dic.items():
        same_p_list = []
        same_p_list.append(ges1)
        for ges2, p2 in gesture_p_dic.items():
            if p1 == p2 and ges1 != ges2:
                same_p_list.append(ges2)
        same_p_clustering[p1] = same_p_list
    print(same_p_clustering)

    for i in range(0, num_latent_semantic):
        if i not in same_p_clustering:
            same_p_clustering[i] = []

    for p, gesture in same_p_clustering.items():
        with open(gestures_folder + "/task4b.csv", 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow([p])
            wr.writerow(gesture)

def task4c(gestures_folder, vector_model, user_option, k, p, r):
    """

    :param gestures_folder:     Given gestuers database
    :param vector_model:        TF or TFIDF: 1  2
    :param user_option:         1,2,3,4,5,6,7
    :param k:                   top k-dimension
    :param p:                   top p-dimension
    :param r:                   resolution
    :return:
    """
    print("Performing task 4c")
    output_vector_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_vector_folder):
        print("execute task0b first")
        return
    list_words, list_files, raw_data = dataMatrix(output_vector_folder, vector_model)

    # sub problem 1:
    gestures_similarity_matrix = generate_gesture_matrix(user_option, raw_data, list_files, gestures_folder, r, list_words)

    model = KMean(n_clusters=p, dist_table=gestures_similarity_matrix, max_iter=20)
    centers, members, costs, tot_cost, dist_mat = model.fit(len(gestures_similarity_matrix), plotit=False, verbose=True)
    final_groups = classify(centers, members)
    for g in sorted(final_groups):
        print("group = {}, member: {}".format(g, final_groups[g]))

def task4d(gestures_folder, vector_model, user_option, k, p, r):
    """

    :param gestures_folder:     Given gestuers database
    :param vector_model:        TF or TFIDF: 1  2
    :param user_option:         1,2,3,4,5,6,7
    :param k:                   top k-dimension
    :param p:                   top p-dimension
    :param r:                   resolution
    :return:
    """
    print("Performing task 4d")
    output_vector_folder = gestures_folder + "/output_vectors"
    if not os.path.exists(output_vector_folder):
        print("execute task0b first")
        return
    list_words, list_files, raw_data = dataMatrix(output_vector_folder, vector_model)

    # sub problem 1:
    gestures_similarity_matrix = generate_gesture_matrix(user_option, raw_data, list_files, gestures_folder, r, list_words)

    W_max = np.amax(gestures_similarity_matrix, 1)
    W = gestures_similarity_matrix / W_max
    D = np.diag(W.sum(axis=1))
    L = D - W
    model = KMean(n_clusters=p, dist_table=L, max_iter=30)
    centers, members, costs, tot_cost, dist_mat = model.fit(len(gestures_similarity_matrix), plotit=False, verbose=True)
    final_groups = classify(centers, members)
    for g in sorted(final_groups):
        print("group = {}, member: {}".format(g, final_groups[g]))


def _get_init_centers(n_clusters, n_samples):
    '''return random points as initial centers'''
    init_ids = []
    while len(init_ids) < n_clusters:
        _ = np.random.randint(0,n_samples)
        if not _ in init_ids:
            init_ids.append(_)
    return init_ids


def _get_distance(index1, index2, distance_table):
    '''example distance function'''
    return 1-distance_table[index1,index2]


def _get_cost(X, center_ids, dist_func, distance_table):
    '''return total cost and cost of each cluster'''
    st = time.time()
    dist_mat = np.zeros((X,len(center_ids)))
    # compute distance matrix
    for j in range(len(center_ids)):
        for i in range(X):
            dist_mat[i,j] = dist_func(i, center_ids[j], distance_table)

    #print 'cost ', -st+time.time()
    mask = np.argmin(dist_mat,axis=1)
    members = np.zeros(X)
    costs = np.zeros(len(center_ids))
    for i in range(len(center_ids)):
        mem_id = np.where(mask==i)
        members[mem_id] = i
        costs[i] = np.sum(dist_mat[mem_id,i])
    return members, costs, np.sum(costs), dist_mat


def _KMean_run(X, n_clusters, dist_func, dist_table, max_iter=1000, tol=0.001, verbose=True):
    '''run algorithm return centers, members, and etc.'''
    # Get initial centers
    n_samples = X
    init_ids = _get_init_centers(n_clusters,n_samples)
    '''
    if verbose:
        print( 'Initial centers are ', init_ids)
    '''
    centers = init_ids
    members, costs, tot_cost, dist_mat = _get_cost(X, init_ids,dist_func, dist_table)
    cc,SWAPED = 0, True
    while True:
        SWAPED = False
        #print("current iter: ", cc)
        for i in range(n_samples):
            if not i in centers:
                for j in range(len(centers)):
                    centers_ = deepcopy(centers)
                    centers_[j] = i
                    members_, costs_, tot_cost_, dist_mat_ = _get_cost(X, centers_,dist_func, dist_table)
                    if tot_cost_-tot_cost < tol:
                        members, costs, tot_cost, dist_mat = members_, costs_, tot_cost_, dist_mat_
                        centers = centers_
                        SWAPED = True
                        '''
                        if verbose:
                            print('Change centers to ', sorted(centers))
                        '''
        if cc > max_iter:
            if verbose:
                print('End Searching by reaching maximum iteration', max_iter)
            break
        if not SWAPED:
            if verbose:
                print('End Searching by no swaps')
            break
        cc += 1
    return centers, members, costs, tot_cost, dist_mat

def classify(centers, classification):
    classes = {}
    for i in range(len(classification)):
        classes[centers[int(classification[i])]] = classes.get(centers[int(classification[i])], list())
        classes[centers[int(classification[i])]].append(i)
    return classes

class KMean(object):
    '''
    Main API of KMean Clustering

    Parameters
    --------
        n_clusters: number of clusters
        dist_func : distance function
        max_iter: maximum number of iterations
        tol: tolerance

    Attributes
    --------
        labels_    :  cluster labels for each data item
        centers_   :  cluster centers id
        costs_     :  array of costs for each cluster
        n_iter_    :  number of iterations for the best trail

    Methods
    -------
        fit(X): fit the model
            - X: 2-D numpy array, size = (n_sample, n_features)

        predict(X): predict cluster id given a test dataset.
    '''
    def __init__(self, n_clusters, dist_table, dist_func=_get_distance, max_iter=10000, tol=0.0001):
        self.n_clusters = n_clusters
        self.dist_table = dist_table
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, plotit=True, verbose=True):
        return _KMean_run(X, self.n_clusters, self.dist_func, self.dist_table, max_iter=self.max_iter, tol=self.tol,verbose=verbose)

    def predict(self,X):
        raise NotImplementedError()

if __name__ == '__main__':
    while True:
        option = input("Please choose following command:\n"
                       "\ttask 0a: 0a\n"
                       "\ttask 0b: 0b\n"
                       "\ttask  1:  1\n"
                       "\ttask  2:  2\n"
                       "\ttask 3a: 3a\n"
                       "\ttask 3b: 3b\n"
                       "\ttask 4a: 4a\n"
                       "\ttask 4b: 4b\n"
                       "\ttask 4c: 4c\n"
                       "\ttask 4d: 4d\n"
                       "\t   quit:  q\n"
                       "Your Option is: ")
        if option == "0a":
            # ./data
            gestures_folder = input("Please enter the path of gestures folder:\t")
            w = int(input("Please enter a window length:\t"))
            s = int(input("Please enter shift length:\t"))
            r = int(input("Please enter a resolution:\t"))
            task0a(gestures_folder, w, s, r)
            print("Task0a is completed\n\n")
        if option == "0b":
            gestures_folder = input("Please enter the path of gestures folder:\t")
            task0b(gestures_folder)
            print("Task0b is completed\n\n")
        if option == "1":
            gestures_folder = input("Please enter the path of gestures folder:\t")
            vector_model = input("Please enter a vector model:\n\ttf: 1\n\ttfidf: 2\n")
            k = input("Please enter the k:\t")
            user_option = input("Please enter one of options listed below:\n\tPCA: 1\n\tSVD: 2\n\tNMF: 3\n\tLDA: 4\n")
            task1(gestures_folder, vector_model, int(k), user_option)
            print("Task1 is completed\n\n")
        if option == "2":
            gestures_folder = input("Please enter the path of gestures folder:\t")
            filename = input("Please enter filename (without suffix):\t")
            vector_model = input("Please enter a vector model:\n\ttf: 1\n\ttfidf: 2\n")
            user_option = input("Please enter one of options listed below:\n\tDot Product: 1\n\tPCA: 2\n\tSVD: "
                                "3\n\tNMF: 4\n\tLDA: 5\n\tEdit Distance: 6\n\tDTW: 7\n")
            k = 0
            if user_option == "2" or user_option == "3" or user_option == "4" or user_option == "5":
                k = input("Please enter the k:\t")
            r = int(input("Please enter a resolution:\t"))
            task2(gestures_folder, filename, vector_model, user_option, int(k), r)
            print("Task2 is completed\n\n")
        if option == "3a":
            gestures_folder = input("Please enter the path of gestures folder:\t")
            vector_model = input("Please enter a vector model:\n\ttf: 1\n\ttfidf: 2\n")
            user_option = input("Please enter one of options listed below:\n\tDot Product: 1\n\tPCA: 2\n\tSVD: "
                                "3\n\tNMF: 4\n\tLDA: 5\n\tEdit Distance: 6\n\tDTW: 7\n")
            k = 0
            if user_option == "2" or user_option == "3" or user_option == "4" or user_option == "5":
                k = input("Please enter the k:\t")
            p = input("Please enter the p:\t")
            r = int(input("Please enter a resolution:\t"))
            task3a(gestures_folder, vector_model, user_option, int(k), int(p), r)
            print("Task3a is completed\n\n")
        if option == "3b":
            gestures_folder = input("Please enter the path of gestures folder:\t")
            vector_model = input("Please enter a vector model:\n\ttf: 1\n\ttfidf: 2\n")
            user_option = input("Please enter one of options listed below:\n\tDot Product: 1\n\tPCA: 2\n\tSVD: "
                                "3\n\tNMF: 4\n\tLDA: 5\n\tEdit Distance: 6\n\tDTW: 7\n")
            k = 0
            if user_option == "2" or user_option == "3" or user_option == "4" or user_option == "5":
                k = input("Please enter the k:\t")
            p = input("Please enter the p:\t")
            r = int(input("Please enter a resolution:\t"))
            task3b(gestures_folder, vector_model, user_option, int(k), int(p), r)
            print("Task3b is completed\n\n")
        if option == "4a":
            gestures_folder = input("Please enter the path of gestures folder:\t")
            task4a(gestures_folder)
            print("Task4a is completed\n\n")
        if option == "4b":
            gestures_folder = input("Please enter the path of gestures folder:\t")
            task4b(gestures_folder)
            print("Task4b is completed\n\n")
        if option == "4c":
            print("Performing task 4c")
            gestures_folder = input("Please enter the path of gestures folder:\t")
            vector_model = input("Please enter a vector model:\n\ttf: 1\n\ttfidf: 2\n")
            user_option = input("Please enter one of options listed below:\n\tDot Product: 1\n\tPCA: 2\n\tSVD: "
                                "3\n\tNMF: 4\n\tLDA: 5\n\tEdit Distance: 6\n\tDTW: 7\n")
            k = 0
            if user_option == "2" or user_option == "3" or user_option == "4" or user_option == "5":
                k = input("Please enter the k:\t")
            p = input("Please enter the p:\t")
            r = int(input("Please enter a resolution:\t"))
            task4c(gestures_folder, vector_model, user_option, int(k), int(p), r)
            print("Task4c is completed\n\n")

        if option == "4d":
            print("Performing task 4d")
            gestures_folder = input("Please enter the path of gestures folder:\t")
            vector_model = input("Please enter a vector model:\n\ttf: 1\n\ttfidf: 2\n")
            user_option = input("Please enter one of options listed below:\n\tDot Product: 1\n\tPCA: 2\n\tSVD: "
                                "3\n\tNMF: 4\n\tLDA: 5\n\tEdit Distance: 6\n\tDTW: 7\n")
            k = 0
            if user_option == "2" or user_option == "3" or user_option == "4" or user_option == "5":
                k = input("Please enter the k:\t")
            p = input("Please enter the p:\t")
            r = int(input("Please enter a resolution:\t"))
            task4d(gestures_folder, vector_model, user_option, int(k), int(p), r)
            print("Task4d is completed\n\n")
        if option == "q":
            print("Quitting the program\n\n")
            break

        if option == "t":
            for i in range(7):
                for y in range(2):
                    if 1 <= i <= 4:
                        task1("./3_class_gesture_data", str(y+1), 4, str(i))
                    task2("./3_class_gesture_data", "1", str(y+1), str(i+1), 4, 3)
