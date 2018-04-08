import pandas as pd
import numpy as np
import random
import math
from knn import Knn

class IrisLDA:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' else (1 if x == 'Iris-versicolor' else 2))
        self.irisdata = df.astype(float).values.tolist()
        self.number_of_features = len(self.irisdata[0]) - 1
        self.train_data = []
        self.train_data_class = {}
        self.test_data = []
        self.means = {}

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

    def getMeansForEachClass(self, train_data):
        num_items = {}
        for train in train_data:
            if train[-1] not in self.means:
                self.means.update({train[-1]: train[:-1]})
                num_items.update({train[-1]: 1.})
                self.train_data_class.update({train[-1]: [train[:-1]]})
            else:
                for i in range(len(train)-1):
                    self.means[train[-1]][i] += train[i]
                num_items[train[-1]] += 1
                self.train_data_class[train[-1]].append(train[:-1])
        for key in self.means:
            for i in range(len(self.means[key])):
                self.means[key][i] /= num_items[key]
        # print (self.means)
        # print (self.train_data_class)
    def getScatterMatrices(self):
        # calculate Sw
        self.s_w = np.zeros((self.number_of_features, self.number_of_features))
        # loop for number of classes
        for key in self.train_data_class:
            s_i = np.zeros((self.number_of_features, self.number_of_features))
            mean_c = np.reshape(self.means[key],(self.number_of_features, 1))
            # loop for nember of observations in the class
            for ele in self.train_data_class[key]:
                # dot product
                x_i = np.reshape(ele,(self.number_of_features, 1))
                s_i += (x_i - mean_c).dot((x_i - mean_c).T)
                # for i in len(ele):
            self.s_w += s_i
        # print ('Scatter Within classes (Sb): \n', self.s_w)

        # calculate Sb
        self.s_b = np.zeros((self.number_of_features, self.number_of_features))
        # get overall mean
        overall_mean = np.zeros((self.number_of_features, 1))
        num_class = 0
        for key in self.train_data_class:
            overall_mean += np.reshape(self.means[key],(self.number_of_features, 1))
            num_class += 1
        overall_mean /= num_class
        # print (overall_mean)
        # Loop for class
        for key in self.train_data_class:
            n = len(self.train_data_class[key])
            mean_i = np.reshape(self.means[key], (self.number_of_features, 1))
            self.s_b += n * (mean_i - overall_mean).dot((mean_i - overall_mean).T)
        # print ('Scatter Between classes (Sb): \n', self.s_b)

    def getTranformMatrixW(self,  k=2):
        self.w = np.zeros((self.number_of_features, k))
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(self.s_w).dot(self.s_b))
        eig_vecs = eig_vecs.real
        # print (eig_vals)
        # print (eig_vecs)
        eig_vals_sorted_index = sorted(range(len(eig_vals)),key=lambda x:eig_vals[x], reverse=True)
        # print (eig_vals_sorted_index)
        for i in range(k):
            for j in range(self.number_of_features):
                self.w[j][i] = eig_vecs[j][eig_vals_sorted_index[i]]
        # print (self.w)

    def getProjectedData(self, data):
        data_z = []
        for features in data:
            dot_result = np.dot(np.transpose(self.w), features[:-1]).tolist()
            dot_result.append(features[-1])
            data_z.append(dot_result)
        # print (data_z)
        return data_z

def loopLdaKnn(loop=1):
    accuracy = 0
    for i in range(loop):
        iris_data = IrisLDA('iris_data_set/iris.data')
        iris_data.randomSplit(35)
        iris_data.getMeansForEachClass(iris_data.train_data)
        iris_data.getScatterMatrices()
        iris_data.getTranformMatrixW()
        new_train_data = iris_data.getProjectedData(iris_data.train_data)
        new_test_data = iris_data.getProjectedData(iris_data.test_data)
        knn = Knn()
        accuracy += knn.kNearestNeighbors(new_train_data, new_test_data)
    return accuracy/loop

if __name__ == "__main__":
    print ("Accuracy: ", format(loopLdaKnn(loop=10), ".3f"))
