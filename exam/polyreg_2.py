import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("../data/auto-mpg.csv")

df.dropna(inplace=True)

x = df[["displacement"]]
y = df['mpg']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

result = []
for degree in range(1,6):
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_poly,y_train)
    y_poly_pred = model.predict(x_test_poly)
    
    mse = mean_squared_error(y_test,y_poly_pred)
    r2 = r2_score(y_test,y_poly_pred)
    result.append([degree,mse,r2])


    #For plotting purpose
    x_plot = np.linspace(
        x['displacement'].min(),
        x['displacement'].max(),
        300).reshape(-1,1)
    x_plot_poly = poly.transform(x_plot)
    y_plot = model.predict(x_plot_poly)

    plt.scatter(x,y,color="blue",label="actual data")
    plt.plot(x_plot,y_plot,color='red',label=f"Regression with degree {degree}")
    plt.title(f"Degree {degree}")
    plt.legend()
    plt.show()

    
for i in result:
    print("degres:",i[0])
    print("Mse:",i[1])
    print("R2:",i[2])
    print("--------------")