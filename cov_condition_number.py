import pandas as pd
import numpy as np
from numpy import linalg as LA

class irisClass():
    def __init__(self):
        self.data = []
        self.cov_matrix = np.zeros((4, 4))
        self.mean = []

    def calMean(self):
        if self.data:
            for i in range(len(self.data)):
                for j in range(4):
                    if len(self.mean) < 4:
                        self.mean.append(self.data[i][j])
                    else:
                        self.mean[j] += self.data[i][j]
            for i in range(len(self.mean)):
                self.mean[i] = self.mean[i] / len(self.data)
            # print (self.mean)
        else:
            raise Exception('No data loaded in this class!')

    def calCovarianceMatrix(self):
        if self.mean:
            number_of_observation = len(self.data)
            number_of_feature = len(self.data[0])
            # print (number_of_observation, ', ', number_of_feature)
            for i in range(number_of_feature):
                for j in range(number_of_feature):
                    result = 0
                    for k in range(number_of_observation):
                        result += (self.data[k][i] - self.mean[i])*(self.data[k][j] - self.mean[j])
                    self.cov_matrix[i][j] = result / number_of_observation
        else:
            raise Exception('No data loaded or not calculate mean in this class yet!')

    def getConditionNumber(self):
        w, v = LA.eig(self.cov_matrix)
        lambda_min = 999999
        lambda_max = -999999
        for i in range(len(w)):
            if w[i] > lambda_max:
                lambda_max =  w[i]
            if w[i] < lambda_min:
                lambda_min = w[i]
        return abs(lambda_max/lambda_min)
# End class

def loadData(file_name, setosa_data, versicolor_data, virginica_data):
    df = pd.read_csv(file_name)
    print ('Loaded {} items'.format(len(df)))
    for i in range(len(df)):
        data_list = df.iloc[i].values.tolist()
        if data_list[-1] == 'Iris-setosa':
            setosa_data.append(data_list[:-1])
        elif data_list[-1] == 'Iris-versicolor':
            versicolor_data.append(data_list[:-1])
        else:
            virginica_data.append(data_list[:-1])

def calMeans():
    setosa_class.calMean()
    versicolor_class.calMean()
    virginica_class.calMean()
    print ('####### Means #######')
    print ('Setosa: \t', setosa_class.mean)
    print ('Setosa (Library):', np.matrix(setosa_class.data).mean(0))
    print ('Versicolor: \t', versicolor_class.mean)
    print ('Setosa (Library):', np.matrix(versicolor_class.data).mean(0))
    print ('Virginica: \t', virginica_class.mean)
    print ('Setosa (Library):', np.matrix(virginica_class.data).mean(0))

def calCovarianceMatrices():
    setosa_class.calCovarianceMatrix()
    versicolor_class.calCovarianceMatrix()
    virginica_class.calCovarianceMatrix()
    print ('####### Covariance Matrix #######')
    print ('Setosa: \n', setosa_class.cov_matrix)
    print ('Setosa (Library): \n', np.cov(np.transpose(setosa_class.data)))
    print ('Versicolor: \n', versicolor_class.cov_matrix)
    print ('Versicolor (Library): \n', np.cov(np.transpose(versicolor_class.data)))
    print ('Virginica: \n', virginica_class.cov_matrix)
    print ('Virginica (Library): \n', np.cov(np.transpose(virginica_class.data)))

def calConditionNumber():
    print ('####### Condition Number #######')
    print ('Setosa: ', setosa_class.getConditionNumber())
    print ('Versicolor: ', versicolor_class.getConditionNumber())
    print ('Virginica: ', virginica_class.getConditionNumber())

if __name__ == "__main__":
    setosa_class = irisClass()
    versicolor_class = irisClass()
    virginica_class = irisClass()
    loadData('iris_data_set/iris.data', setosa_class.data, versicolor_class.data, virginica_class.data)
    print ('Number of setosa: ', len(setosa_class.data))
    print ('Number of versicolor: ', len(versicolor_class.data))
    print ('Number of virginica: ', len(virginica_class.data))
    calMeans()
    calCovarianceMatrices()
    calConditionNumber()
