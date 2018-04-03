# PCA ussing excting library: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
import pandas as pd
import numpy as np
import random
import math

class IrisDataset:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' else (1 if x == 'Iris-versicolor' else 2))
        self.irisdata = df.astype(float).values.tolist()
        self.train_data = []
        self.test_data = []
        self.train_mean = []
        self.train_standard_deviation = []

    def randomSplit(self, number_of_train_for_eachclass=35):
        random.shuffle(self.irisdata)
        num_of_setosa_in_train = 0
        num_of_versicolor_in_train = 0
        num_of_virginica_in_train = 0
        for data in self.irisdata:
            if data[-1] == 0:
                if num_of_setosa_in_train < number_of_train_for_eachclass:
                    self.train_data.append(data)
                    num_of_setosa_in_train += 1
                else:
                    self.test_data.append(data)
            elif data[-1] == 1:
                if num_of_versicolor_in_train < number_of_train_for_eachclass:
                    self.train_data.append(data)
                    num_of_versicolor_in_train += 1
                else:
                    self.test_data.append(data)
            else:
                if num_of_virginica_in_train < number_of_train_for_eachclass:
                    self.train_data.append(data)
                    num_of_virginica_in_train += 1
                else:
                    self.test_data.append(data)
        # print ('Number of training data', len(self.train_data))
        # print ('Number of testing data', len(self.test_data))

    def calTrainMeanSd(self, train):
        # calculate mean
        for data in train:
            if len(self.train_mean) == 0:
                for i in range(len(data)-1):
                    self.train_mean.append(data[i])
            else:
                for i in range(len(data)-1):
                    self.train_mean[i] += data[i]
        for i in range(len(data)-1):
            self.train_mean[i] = self.train_mean[i] / len(train)
        print (self.train_mean)
        # calculate standard deviation
        for data in train:
            if len(self.train_standard_deviation) == 0:
                for i in range(len(data)-1):
                    self.train_standard_deviation.append(pow(data[i]-self.train_mean[i], 2))
            else:
                for i in range(len(data)-1):
                    self.train_standard_deviation[i] += pow(data[i]-self.train_mean[i], 2)
        for i in range(len(data)-1):
            self.train_standard_deviation[i] = math.sqrt(self.train_standard_deviation[i] / len(train))
        print (self.train_standard_deviation)

    def zScoreNormalize(self, dataset):
        # Normalize train_set
        # print (dataset)
        for data in dataset:
            for i in range(len(data)-1):
                data[i] = (data[i] - self.train_mean[i]) / self.train_standard_deviation[i]
        # print (dataset)


if __name__ == "__main__":
    iris_data = IrisDataset('iris_data_set/iris.data')
    iris_data.randomSplit(35)
    iris_data.calTrainMeanSd(iris_data.train_data)
    iris_data.zScoreNormalize(iris_data.train_data)
