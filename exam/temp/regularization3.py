from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = load_diabetes()

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
linear_pred = model.predict(x_test)
linear_mse = mean_squared_error(y_test,linear_pred)
linear_r2 = r2_score(y_test,linear_pred)

print("----Linear Regression----")
print("Mse:",linear_mse)
print("R2:",linear_r2)

#ridge
ridge_pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("ridge",Ridge())
])
ridge_params = {
    "ridge__alpha" : [0.01,0.1,1,10,100]
}

ridge_grid = GridSearchCV(
    ridge_pipeline,
    ridge_params,
    cv = 5,
    scoring="neg_mean_squared_error"
)
ridge_grid.fit(x_train,y_train)
ridge_pred = ridge_grid.predict(x_test)
ridge_mse = mean_squared_error(y_test,ridge_pred)
ridge_r2 = r2_score(y_test,ridge_pred)

print("----Ridge regression----")
print("Mse:",ridge_mse)
print("R2:",ridge_r2)

#Lasso
lasso_pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("lasso",Lasso(max_iter=1000))
])

lasso_params = {
    "lasso__alpha" : [0.01,0.1,1,10,100]
}

lasso_grid = GridSearchCV(
    lasso_pipeline,
    lasso_params,
    cv=5,
    scoring="neg_mean_squared_error"
)
lasso_grid.fit(x_train,y_train)
lasso_pred = lasso_grid.predict(x_test)
lasso_mse = mean_squared_error(y_test,lasso_pred)
lasso_r2 = r2_score(y_test,lasso_pred)

print("----Lasso----")
print("Mse:",lasso_mse)
print("R2:",lasso_r2)

