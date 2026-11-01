from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

model = LogisticRegression(penalty=None,max_iter=1000)
model.fit(x_train,y_train)
pred = model.predict(x_test)
accuracy = accuracy_score(y_test,pred)

print("MLE--------------")
print("Accuracy score:",accuracy)

#now l2
l2_model = LogisticRegression(
    penalty='l2',
    C=1,
    max_iter=1000
)
l2_model.fit(x_train,y_train)
l2_pred = l2_model.predict(x_test)
l2_accuracy = accuracy_score(y_test,l2_pred)

print("L2 MAP-----------------")
print("Accuracy:",l2_accuracy)

#L1
l1_model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1,
    max_iter=1000
)
l1_model.fit(x_train,y_train)
l1_pred = l1_model.predict(x_test)
l1_accuracy = accuracy_score(y_test,l1_pred)
print("L1 MAP-----------")
print("Accuracy:",l1_accuracy)

