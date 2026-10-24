import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/housing.csv")
df.dropna(inplace=True)

df["average_house"] = df["total_rooms"]/df["households"]
x = df["average_house"].values
y = df["median_house_value"].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#normalize it
mean = x_train.mean()
std = x_train.std()

x_train_norm = (x_train - mean)/std
x_test_norm = (x_test - mean)/std

print("Training mean:",mean)
print("Training std",std)


b0 = 0
b1 = 0
iteration = 1000
learning_rate = 0.01
n = len(x_train_norm)
cost = []

for i in range(iteration):
    y_pred = b0+b1*x_train_norm
    mse = np.mean((y_train - y_pred)**2)
    cost.append(mse)
    #partial derivatives
    db0 = (-2/n)*np.sum(y_train - y_pred)
    db1 = (-2/n)*np.sum(x_train_norm*(y_train-y_pred))
    #update
    b0 = b0 - learning_rate*db0
    b1 = b1 - learning_rate*db1
y_pred_gd = b0+b1*x_test_norm

#gradient graph
plt.scatter(x_test_norm,y_test,color='blue',label='actual data')
plt.plot(x_test_norm,y_pred_gd,color='red',label='regression line')
plt.xlabel("Average number of rooms")
plt.ylabel("median house value")
plt.title("Gradient Descent Graph")
plt.legend()
plt.show()

#cost convergence graph
plt.plot(range(iteration),cost,color="green")
plt.xlabel("Iteration")
plt.ylabel("cost")
plt.legend()
plt.show()

print("----Gradient desent----")
print("MSE:",mean_squared_error(y_test,y_pred_gd))
print("R2 score:",r2_score(y_test,y_pred_gd))

