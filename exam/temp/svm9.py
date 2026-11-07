import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn import svm

iris = load_iris()

x = iris.data[:,:2]
y = np.where(iris.target == 0,1,-1) #where(condition,value_if_true,value_if_false)

model = svm.SVC(kernel='linear',C=1)

model.fit(x,y)

plt.scatter(
    x[:,0],#All the rows and first column
    x[:,1],#All the rows and second column
    c = y,
    cmap='coolwarm'
)

ax = plt.gca() #get the current axis

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx,yy = np.meshgrid(
    np.linspace(xlim[0],xlim[1],100),#creates 100 equally spaced values from min to max
    np.linspace(ylim[0],ylim[1],100)
)


z =  model.decision_function(
    np.c_[xx.ravel(),yy.ravel()]
).reshape(xx.shape)

ax.contour(
    xx,yy,z,
    label=[-1,0,1],
    colors=['blue','black','red'],
    linestyle=['--','-','--']
)

ax.scatter(
    model.support_vectors_[:,0],
    model.support_vectors_[:,1],
    s=100,
    facecolor='none',
    edgecolors='green'
)

plt.xlabel("Sepal length")
plt.ylabel("sepal width")

plt.title("Linear svm on iris dataset")
plt.show()