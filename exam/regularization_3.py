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


#We are doing basically that, a pipeline object created that -
#combines preprocessing steps and a machine learning model into a single workflow
#Then creates the prameter dict of list
#Then pass these values to `GridSearchCV` to automate the process of finding the best `alpha` value using 5-fold cross-validation.



#Ridge pipeline
ridge_pipeline = Pipeline([ #The object
    ("scaler",StandardScaler()),
    ("ridge" ,Ridge()) #ATTENTION -> In here we use "ridge" as the step name So use that same name in the params too
])

ridge_params = {
    "ridge__alpha" :[0.01,0.1,1,10,100] #In here (xx__alpha) should match the step name.In my case it is "ridge" so it will become "ridge__alpha"
    #Also use dounle Underscore (__)
}

ridge_grid = GridSearchCV(  #Used to automate the model and find best value
    ridge_pipeline, #object
    ridge_params, #params
    cv=5, #Number of cross-validation folds(CV)
    scoring='neg_mean_squared_error' #Metric used to compare models 
    #Negative used here beacuse the gridSearchCV follows greater value is better but for MSE it is reverse
)

ridge_grid.fit(x_train,y_train)
ridge_prediction = ridge_grid.predict(x_test)

ridge_mse = mean_squared_error(y_test,ridge_prediction)
ridge_r2 = r2_score(y_test,ridge_prediction)

print("Ridge mse:",ridge_mse)
print("Ridge R2:",ridge_r2)