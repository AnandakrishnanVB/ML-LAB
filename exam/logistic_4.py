# Logistic Regression is a classification algorithm used to predict categories.

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,random_state=42
)

#Now do the Scaling.
#Scaling makes the uneven values to a smooth range

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Now the logistic regression
model = LogisticRegression(penalty=None,max_iter=1000) #In here use the "penalty=None" for the MLE without regularization.
#by default it uses the L2 regularization