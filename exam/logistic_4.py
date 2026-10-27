# Logistic Regression is a classification algorithm used to predict categories.
#MLE (Maximum Likelihood Estimation) estimates model parameters by finding the values that make the observed data most likely.
#MAP (Maximum A Posteriori estimation) estimates model parameters using the observed data and prior information about the parameters.

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

model.fit(x_train,y_train)
mle_pred = model.predict(x_test)

mle_accuracy = accuracy_score(y_test,mle_pred)

l2_model = LogisticRegression(
    penalty='l2',
    C=1,
    max_iter=1000
)

l2_model.fit(x_train,y_train)
l2_pred = l2_model.predict(x_test)

l2_accuracy = accuracy_score(y_test,l2_pred)

l1_model = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=1,
    max_iter=1000
)

l1_model.fit(x_train,y_train)
l1_pred = l1_model.predict(x_test)

l1_accuracy = accuracy_score(y_test,l1_pred)


print("----MODEL PERFORMANCE----")
print("mle accuracy:",mle_accuracy)
print("L2 accuracy",l2_accuracy)
print("L1 accuracy:",l1_accuracy)

print("----PARAMETER ESTIMATES----")
print("MLE coefficient:",model.coef_)
print("L2 Coefficent:",l2_model.coef_)
print("L1 coefficient:",l1_model.coef_)

