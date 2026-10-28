import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("../data/housing.csv")
df.dropna(inplace=True)

df["average_rooms"] = df["total_rooms"] / df["households"]
x = df["average_rooms"].values
y = df["median_house_value"].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Geadient desent method

#First normalize it
mean = x_train.mean()
std = x_train.std()

x_train_norm = (x_train - mean) / std
x_test_norm = (x_test - mean) / std

b0 = 0
b1 = 0
learning_rate = 0.01
iteration = 1000
n = len(x_train_norm)
cost = []
for i in range(iteration):
    y_pred = b0+b1*x_train_norm
    mse = np.mean((y_train-y_pred)**2)
    cost.append(mse)
    db0 = (-2/n)*np.sum(y_train-y_pred)
    db1 = (-2/n)*np.sum(x_train_norm*(y_train-y_pred))
    b0 = b0 - learning_rate*db0
    b1 = b1 - learning_rate*db1

y_pred_gd = b0+b1*x_test_norm

plt.scatter(x_test_norm,y_test,color="blue",label="Actual data")
plt.plot(x_test_norm,y_pred_gd,color="red",label="Regression line")
plt.xlabel("Average no of rooms")
plt.ylabel("house value")
plt.title("Gradient desent")
plt.legend()
plt.show()

plt.plot(range(1000),cost,color="red")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.legend()
plt.show()



model = LinearRegression()
model.fit(x_train_norm.reshape(-1,1),y_train)
pred = model.predict(x_test_norm.reshape(-1,1))

linear_mse = mean_squared_error(y_test,pred)
linear_r2 = r2_score(y_test,pred)
print("MSE:",linear_mse)
print("R2:",linear_r2)

m = model.coef_[0]
c = model.intercept_

plt.scatter(x_test,y_test,color="blue",label="actual data")
x_line = np.sort(x_test)
y_line = m*x_line+c
plt.plot(x_line,y_line,color="black",label="regression line")
plt.xlabel("avg no of house")
plt.ylabel("households")
plt.show()