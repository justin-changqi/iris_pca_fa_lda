# PCA ussing excting library: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# Good expend for PCA: https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
import pandas as pd
import numpy as np
import random
import math
from numpy import linalg as LA

class IrisPCA:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' else (1 if x == 'Iris-versicolor' else 2))
        self.irisdata = df.astype(float).values.tolist()
        self.train_data = []
        self.test_data = []
        self.train_mean = []
        self.train_standard_deviation = []
        self.projectionMatrixW = []

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
        # print (self.train_mean)
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
        # print (self.train_standard_deviation)

    def zScoreNormalize(self, dataset):
        for data in dataset:
            for i in range(len(data)-1):
                data[i] = (data[i] - self.train_mean[i]) / self.train_standard_deviation[i]

    def calProjectionMatrixW(self, number_of_conponent=2):
        number_of_observation = len(self.train_data)
        number_of_feature = len(self.train_data[0]) - 1
        train_cov_matrix = np.zeros((number_of_feature, number_of_feature))
        for i in range(number_of_feature):
            for j in range(number_of_feature):
                result = 0
                for k in range(number_of_observation):
                    result += (self.train_data[k][i] - self.train_mean[i])*(self.train_data[k][j] - self.train_mean[j])
                train_cov_matrix[i][j] = result / number_of_observation
        # print (train_cov_matrix)
        # get Eigenvalues and Eigenvector for convariance matrix
        eig_vals, eig_vecs = LA.eig(train_cov_matrix)
        eig_vals_sorted_index = sorted(range(len(eig_vals)),key=lambda x:eig_vals[x], reverse=True)
        # print (eig_vals)
        # print (eig_vecs)
        # print (v_sorted_index)
        for i in range(number_of_feature):
            ele = []
            for j in range(number_of_conponent):
                ele.append(eig_vecs[i][eig_vals_sorted_index[j]])
            self.projectionMatrixW.append(ele)
        # print (np.array(self.projectionMatrixW))

    def getProjectedData(self, data):
        data_z = []
        for features in data:
            dot_result = np.dot(np.transpose(self.projectionMatrixW), features[:-1]).tolist()
            dot_result.append(features[-1])
            data_z.append(dot_result)
        # print (data_z)
        return data_z

class Knn:
    def __init__(self):
        pass
    def kNearestNeighbors(self, train, test, k=3):
        correct = 0
        for data in test:
            if data[-1] == self.getVoteResult(train, data[:-1], k):
                correct += 1
        # return format(correct/len(test), '.3f')
        return correct/len(test)

    def getVoteResult(self, train, predict, k):
        distances = []
        for data in train:
            squre_sum = 0
            for i in range(len(data)-1):
                squre_sum += pow(data[i] - predict[i], 2)
            distances.append([math.sqrt(squre_sum), data[-1]])
        # print (distances)
        # sort with distance
        distances_sorted = []
        for ele in distances:
            if len(distances_sorted) == 0:
                distances_sorted.append(ele)
            else:
                index = 0
                for i in range(len(distances_sorted)):
                    if ele[0] > distances_sorted[index][0]:
                        index = i
                    else:
                        break
                distances_sorted.insert(index, ele)
        # print (np.array(distances_sorted))
        vote = []
        for data in distances_sorted[:k]:
            index = -1
            for i in range(len(vote)):
                if vote[i][0] == data[1]:
                    index = i
                    break
            if index == -1:
                vote.append([data[1], 1])
            else:
                vote[index][1] += 1
        max_class = [-1, -1]    # [class, number of vote]
        for data in vote:
            if data[1] > max_class[1]:
                max_class[0] = data[0]
                max_class[1] = data[1]
        return max_class[0]


def loopPcaKnn(loop=1):
    accuracy = 0
    for i in range(loop):
        iris_data = IrisPCA('iris_data_set/iris.data')
        iris_data.randomSplit(35)
        # get means and Standard deviation for training data
        iris_data.calTrainMeanSd(iris_data.train_data)
        # apply Z score normalize for training data
        iris_data.zScoreNormalize(iris_data.train_data)
        # get Projection Matrix W
        iris_data.calProjectionMatrixW(number_of_conponent=2)
        # apply Z score normalize for testing data
        iris_data.zScoreNormalize(iris_data.test_data)
        new_train_data = iris_data.getProjectedData(iris_data.train_data)
        new_test_data = iris_data.getProjectedData(iris_data.test_data)
        knn = Knn()
        accuracy += knn.kNearestNeighbors(new_train_data, new_test_data)
    return accuracy/loop


if __name__ == "__main__":
    print ("Accuracy: ", format(loopPcaKnn(loop=10), ".3f"))
