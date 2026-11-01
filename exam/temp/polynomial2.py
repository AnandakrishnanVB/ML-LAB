import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("../data/auto-mpg.csv")

x = df[["displacement"]]
y = df["mpg"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
result = []
for degree in range(1,6):
    poly = PolynomialFeatures(degree=degree)
    x_trian_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    model = LinearRegression()
    model.fit(x_trian_poly,y_train)
    pred = model.predict(x_test_poly)

    mse = mean_squared_error(y_test,pred)
    r2 = r2_score(y_test,pred)
    result.append([degree,mse,r2])



    x_plot = np.linspace(
        df['displacement'].min(),
        df['displacement'].max(),300
    ).reshape(-1,1)
    x_plot_poly = poly.transform(x_plot)
    y_plot = model.predict(x_plot_poly)

    plt.scatter(x,y,color="blue",label="actual data")
    plt.plot(x_plot,y_plot,color="red",label=f"Regression degress {degree}")
    plt.title(f"Degree:{degree}")
    plt.xlabel("displacement")
    plt.ylabel("mpg")
    plt.show()
