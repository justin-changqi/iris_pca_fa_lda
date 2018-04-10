import pandas as pd
import numpy as np
import random
import math
from numpy import linalg as LA
from knn import Knn

class IrisFA:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' else (1 if x == 'Iris-versicolor' else 2))
        self.irisdata = df.astype(float).values.tolist()
        self.train_data = []
        self.test_data = []
        self.number_of_features = len(self.irisdata[0]) - 1

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

    def getProjectionMatrixW(self, k=2):
        # calculate covarince matrix for taining set
        num_f = self.number_of_features
        mean_vector = np.zeros(num_f)
        for data in self.train_data:
                mean_vector += np.array(data[:-1])
        mean_vector /= len(self.train_data)
        # print (mean_vector)
        covariance_matrix = np.zeros((num_f, num_f))
        for i in range(num_f):
            for j in range(num_f):
                for data in self.train_data:
                    covariance_matrix[i][j] += (data[i] - mean_vector[i])*(data[j] - mean_vector[j])
                covariance_matrix[i][j] /= len(self.train_data)
        # print (covariance_matrix)
        # print (np.cov(np.transpose(self.train_data)[:-1]))
        eig_vals, eig_vecs = LA.eig(covariance_matrix)
        eig_vals_sorted_index = sorted(range(len(eig_vals)),key=lambda x:eig_vals[x], reverse=True)
        # print ("\nEigenvalues for train covariance matrix: \n", eig_vals)
        # print ("\nEigenvectors for train covariance matrix: \n", eig_vecs)
        # print ("\nSorted Eigenvalues index: ", eig_vals_sorted_index)
        C = np.zeros((num_f, k))
        D_sqrt = np.zeros((k, k))
        for i in range(num_f):
            for j in range(k):
                C[i][j] = eig_vecs[i][j]
        # print ("\nC: \n", C)
        for i in range(k):
            D_sqrt[i][i] = math.sqrt(eig_vals[eig_vals_sorted_index[i]])
        # print ("\nD square root: \n", D_sqrt)
        V = C.dot(D_sqrt)
        # print ("\nV: \n", V)
        self.W = np.linalg.inv(covariance_matrix).dot(V)
        # print ("\nProjection Matrix: \n", self.W)

    def getProjectedData(self, data):
        data_z = []
        for features in data:
            dot_result = np.dot(np.transpose(self.W), features[:-1]).tolist()
            dot_result.append(features[-1])
            data_z.append(dot_result)
        # print (data_z)
        return data_z

def loopFaKnn(loop=1):
    accuracy = 0
    for i in range(loop):
        iris_data = IrisFA('iris_data_set/iris.data')
        iris_data.randomSplit(35)
        iris_data.getProjectionMatrixW(k=2)
        new_train_data = iris_data.getProjectedData(iris_data.train_data)
        new_test_data = iris_data.getProjectedData(iris_data.test_data)
        # print (np.array(iris_data.train_data))
        # print (np.array(new_train_data))
        knn = Knn()
        # print ("Round ",i+1, " 3-NN accuracy: ", format(knn.kNearestNeighbors(new_train_data, new_test_data), ".3f"))
        accuracy += knn.kNearestNeighbors(new_train_data, new_test_data)
    return accuracy/loop

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    print ("Average Accuracy: ", format(loopFaKnn(loop=10), ".3f"))
