class Decision_Tree:
    def __init__(self, column, value, left_branch, right_branch):
        self.column = column
        self.value = value
        self.left_branch = left_branch
        self.right_branch = right_branch


def split(raw_data, column, value):
    left_branch = []
    right_branch = []
    for data in raw_data:
        if data[column] < value:
            left_branch.append(data)
        else:
            right_branch.append(data)
    return left_branch, right_branch


def gini(raw_data):
    labels_count = {}
    for data in raw_data:
        if data[-1] not in labels_count:
            labels_count[data[-1]] = 0
        labels_count[data[-1]] += 1
    impurity = 1
    for label in labels_count:
        impurity -= (labels_count[label] / float(len(raw_data))) ** 2
    return impurity


def information_gain(left_branch, right_branch, impurity):
    p = float(len(left_branch)) / (len(left_branch) + len(right_branch))
    return impurity - p * gini(left_branch) - (1 - p) * gini(right_branch)


def best_split(raw_data):
    best_gain = 0
    best_column = 0
    best_value = 0
    impurity = gini(raw_data)
    for column in range(len(raw_data[0]) - 1):
        for value in set([data[column] for data in raw_data]):
            split_value = split(raw_data, column, value)
            left_branch = split_value[0]
            right_branch = split_value[1]
            if len(left_branch) == 0 or len(right_branch) == 0:
                continue
            gain = information_gain(left_branch, right_branch, impurity)
            if gain >= best_gain:
                best_gain = gain
                best_column = column
                best_value = value
    if best_gain >= 0.0000001:
        split_value = split(raw_data, best_column, best_value)
        left_branch = split_value[0]
        right_branch = split_value[1]
        left_branch = best_split(left_branch)
        right_branch = best_split(right_branch)
        sub_tree = Decision_Tree(best_column, best_value, left_branch, right_branch)
        return sub_tree
    else:
        return raw_data[0][-1]


def fit(raw_data, sub_tree):
    # sub_tree, best_gain = best_split(raw_data)
    if type(sub_tree) == str:
        return sub_tree
    if raw_data[sub_tree.column] < sub_tree.value:
        left_sub_tree = fit(raw_data, sub_tree.left_branch)
        return left_sub_tree
    else:
        right_sub_tree = fit(raw_data, sub_tree.right_branch)
        return right_sub_tree


def decision_tree_train_test(labels_dic, list_files, raw_data):
    training_data = []
    testing_data = []
    testing_file = []
    for i in range(len(list_files)):
        if list_files[i] in labels_dic:
            tmp = []
            for j in range(len(raw_data[0])):
                tmp.append(raw_data[i][j])
            tmp.append(labels_dic[list_files[i]])
            training_data.append(tmp)
        else:
            tmp = []
            for j in range(len(raw_data[0])):
                tmp.append(raw_data[i][j])
            testing_data.append(tmp)
            testing_file.append(list_files[i])
    decision_tree = best_split(training_data)
    for i in range(len(testing_data)):
        tmp = list_files[i]
        tmp1 = fit(testing_data[i], decision_tree)
        print("File name: %s Predicted label: %s" % (tmp, tmp1))
        labels_dic[tmp] = tmp1
    return labels_dic
