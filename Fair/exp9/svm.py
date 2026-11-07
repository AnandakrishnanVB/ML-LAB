
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

# 1. Load Iris dataset
iris = datasets.load_iris()

# Select first two features: Sepal Length and Sepal Width
X = iris.data[:, :2]

# Assign +1 to class 0 and -1 to all other classes
y = np.where(iris.target == 0, 1, -1)

# 2. Train Linear SVM model
model = svm.SVC(kernel='linear', C=1)

model.fit(X, y)

# 3. Plot data points
plt.scatter(
    X[:, 0],#All the rows and first column
    X[:, 1],#All the rows and second column
    c=y,
    cmap='coolwarm'
)

# 4. Create grid for drawing decision boundary
ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(
    np.linspace(xlim[0], xlim[1], 100),
    np.linspace(ylim[0], ylim[1], 100)
)

# 5. Calculate decision values
Z = model.decision_function(
    np.c_[xx.ravel(), yy.ravel()]
).reshape(xx.shape)

# 6. Draw decision boundary and margins
ax.contour(
    xx, yy, Z,
    levels=[-1, 0, 1],
    colors=['blue', 'black', 'blue'],
    linestyles=['--', '-', '--']
)

# 7. Highlight support vectors
ax.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    facecolors='none',
    edgecolors='green'
)

# 8. Add labels and display plot
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Linear SVM on Iris Dataset")

plt.show()
