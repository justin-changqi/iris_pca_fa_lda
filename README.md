# 2018 Spring: Taipei Tech course number 241591 "Machine Learning" homework \#2 source code.
## System Environment
- Ubuntu16.04 LTS
- Python 3.0
- Libraries: Pandas, Numpy, matplotlib

## Install Libraries in Unbuntu for Python3
```
sudo pip3 install pandas
sudo pip3 install numpy
sudo pip3 install matplotlib
```

## Problem 2: Covariance and Condition numbers
For execute the code
```
python3 cov_condition_number.py
```
### Implementation Steps
Iris Dataset format

<!-- x_{n} = \begin{bmatrix}
sepal\ length (x_{n1}), &sepal\ width (x_{n2}),  &petal\ length (x_{n3}),  &petal\ width (x_{n4})
\end{bmatrix} -->
![iris_format_mat](http://latex.codecogs.com/png.latex?%5Cdpi%7B200%7D%20%5Cfn_phv%20%5Ctiny%20x_%7Bn%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20sepal%5C%20length%20%28x_%7Bn1%7D%29%2C%20%26sepal%5C%20width%20%28x_%7Bn2%7D%29%2C%20%26petal%5C%20length%20%28x_%7Bn3%7D%29%2C%20%26petal%5C%20width%20%28x_%7Bn4%7D%29%20%5Cend%7Bbmatrix%7D)

<!-- \forall n=1,...,150 -->
![](http://latex.codecogs.com/png.latex?%5Cdpi%7B200%7D%20%5Cfn_phv%20%5Ctiny%20%5Cforall%20n%3D1%2C...%2C150)
