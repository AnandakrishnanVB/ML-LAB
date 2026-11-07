import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn import svm

iris = load_iris()

x = iris.data[:,:2]
y = np.where(iris.target == 0,1,-1) #where(condition,value_if_true,value_if_false)


