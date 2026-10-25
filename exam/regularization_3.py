from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = load_diabetes()
x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
linear_predict = linear_model.predict(x_test)
linear_mse = mean_squared_error(y_test,linear_predict)
linear_r2 = r2_score(y_test,linear_predict)

print("Linear mse:",linear_mse)
print("Linear r2:",linear_r2)


#Ridge pipeline
ridge_pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("ridge" ,Ridge())
])

ridge_params = {
    "ridge_params" :[0.01,0.1,1,10,100]
}

ridge_grid = GridSearchCV( #It acceps
    ridge_pipeline, #object
    ridge_params, #params
    cv=5, #Number of cross-validation folds(CV)
    scoring='neg_mean_squared_error' #Metric used to compare models 
    #Negative used here beacuse the gridSearchCV follows greater value is better but for MSE it is reverse
)

