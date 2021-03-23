# Gesture-Recognition
A study aimed at realizing remote operations by recognizing gestures to ensure social distancing

## Phase 1

### Introduction

In this project, we assume that there are several documents in a specific directory where each document is a gesture containing time series and sensors. Each document contains the same sensors but different time series and data. Our first goal is to parse each sensor in the given time series and put those data into different bands, and then separate them by a specific window length and finally write into different files. The second goal is to use the terminology of TF and TF-IDF which find the key words in a specific document or the whole document directory. The third goal is to use the generated TF and TF-IDF outputs to map a heat map whose x-axis is time series and y-axis is sensor-ids. The final goal is to compare each document (gesture) with others and find out the top 10 similar ones. We call it gesture comparison.

### Requirements

- Python 3.7
- Pycharm 2.0 (or higher)

### ResultExplanation

- Task1(3,2) gives the output of 66 .wrd.csv files in the Output folder
- Task2() gives the output of vector.txt in the Output folder
- Task3() contains three loops and generates heatmap images which will be saved in the ImgOutput folder
- Task4("test1", "TF_IDF2") means the user selects test1.wrd.csv and TF_IDF2 as input. This can be customized by users.The output will be shown in the folder of SimilarOutput folder. If you want to rerun this, please comment out line 376 - line 391!!!

### Key Words
- Gesture Recognition
- Document Categorization
- Heatmap
- Term Frequency
- Term Frequence Inverse Document Frequency
- Python Dictionary
- File I/O

### Maintainer
- Wei Xin


## Phase 2

This phase is to experiment with dimensionality reduction, unsupervised learning and time series. Feature reduction is done with 4 different models (PCA, SVD, NMF and LDA). 10 most similar gestures are found and ranked. Analysis is done on latent gesture discovery and latent gesture clustering.

# Getting Started

### Installation Prerequisites

* Python 3.7
* gensim             3.8.3
* idna               2.10
* importlib-metadata 2.0.0
* lda                2.0.0
* matlab             0.1
* matplotlib         3.3.2
* numpy              1.19.2
* pandas             1.1.3
* pip                20.2.3
* pyLDAvis           2.1.2
* scikit-learn       0.23.2
* scipy              1.5.2
* copy
* time

### Running the tests

```sh
$ python3 phase2.py
```

### Built With
* PyCharm

This phase is to experiment with dimensionality reduction, unsupervised learning and time series. Feature reduction is done with 4 different models (PCA, SVD, NMF and LDA). 10 most similar gestures are found and ranked. Analysis is done on latent gesture discovery and latent gesture clustering.

## Phase 2

### Installation Prerequisites

* Python 3.7
* gensim             3.8.3
* idna               2.10
* importlib-metadata 2.0.0
* lda                2.0.0
* matlab             0.1
* matplotlib         3.3.2
* numpy              1.19.2
* pandas             1.1.3
* pip                20.2.3
* pyLDAvis           2.1.2
* scikit-learn       0.23.2
* scipy              1.5.2
* copy
* time

### Running the tests

```sh
$ python3 phase2.py
```

### Built With
* PyCharm
