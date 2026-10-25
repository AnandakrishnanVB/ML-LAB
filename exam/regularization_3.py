from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
#Linear Regression predicts continuous values based on input features.
# Ridge Regression uses L2 regularization to reduce overfitting by penalizing large coefficients.
# Lasso Regression uses L1 regularization and can make some coefficients zero to remove unimportant features.
# Regularization helps prevent overfitting by penalizing model complexity.

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# StandardScaler scales features to have a mean of 0 and a standard deviation of 1.
# Pipeline combines preprocessing steps and a machine learning model into a single workflow.



data = load_diabetes()
x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
linear_predict = linear_model.predict(x_test)
linear_mse = mean_squared_error(y_test,linear_predict)
linear_r2 = r2_score(y_test,linear_predict)

print("LINEAR REGRESSION")
print("Linear mse:",linear_mse)
print("Linear r2:",linear_r2)


#We are basically doing that, a pipeline object created that combines-
#preprocessing steps and a machine learning model into a single workflow.
#Then creates the prameter dict of list.
#Then pass these values to `GridSearchCV` to automate the process of finding the best `alpha` value using 5-fold cross-validation.



#Ridge pipeline
ridge_pipeline = Pipeline([ #The object
    ("scaler",StandardScaler()),
    ("ridge" ,Ridge()) #ATTENTION -> In here we use "ridge" as the step name So use that same name in the params too
])
# Alpha controls the strength of regularization; higher alpha generally means stronger regularization.
ridge_params = {
    "ridge__alpha" :[0.01,0.1,1,10,100] #In here (xx__alpha) should match the step name.In my case it is "ridge" so it will become "ridge__alpha"
    #Also use double Underscore (__)
}

# GridSearchCV finds the best hyperparameter by testing different values using cross-validation.
# Cross-validation evaluates a model using different splits of the training data.
ridge_grid = GridSearchCV(  
    ridge_pipeline, #object
    ridge_params, #params
    cv=5, #Number of cross-validation folds(CV)
    scoring='neg_mean_squared_error' #Metric used to compare models 
    #Negative used here beacuse the gridSearchCV follows the rule -> "greater value is better" but for MSE it is reverse
)

ridge_grid.fit(x_train,y_train)
ridge_prediction = ridge_grid.predict(x_test)

ridge_mse = mean_squared_error(y_test,ridge_prediction)
ridge_r2 = r2_score(y_test,ridge_prediction)

print("RIDGE REGRESSION")
print("best alpa",ridge_grid.best_params_['ridge__alpha'])
print("Ridge mse:",ridge_mse)
print("Ridge R2:",ridge_r2)

#Now the lasso
lasso_pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('lasso',Lasso(max_iter=1000)) 
])

lasso_param = {
    "lasso__alpha" : [0.01,0.1,1,10,100]
}

lasso_grid = GridSearchCV(
    lasso_pipeline,
    lasso_param,
    cv=5,
    scoring="neg_mean_squared_error"
)
lasso_grid.fit(x_train,y_train)
lasso_predict = lasso_grid.predict(x_test)
lasso_mse = mean_squared_error(y_test,lasso_predict)
lasso_r2 = r2_score(y_test,lasso_predict)

print("LASSO REGRESSION")
print("Best alpha:",lasso_grid.best_params_['lasso__alpha'])
print("lasso mse:",lasso_mse)
print("Lasso r2:",lasso_r2)