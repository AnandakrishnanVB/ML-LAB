from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#MLE
mle = LogisticRegression(penalty=None,max_iter=1000)
mle.fit(x_train,y_train)
mlepred = mle.predict(x_test)
mle_accu = accuracy_score(y_test,mlepred)

#L2
l2 = LogisticRegression(penalty='l2',max_iter=1000)
l2.fit(x_train,y_train)
l2pred = l2.predict(x_test)
l2_accu = accuracy_score(y_test,l2pred)

#L1
l1 = LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000)
l1.fit(x_train,y_train)
l1pred = l1.predict(x_test)
l1_accu = accuracy_score(y_test,l1pred)

print("MLE")
print("Accuracy:",mle_accu)
print("MAP L2")
print("Accuracy:",l2_accu)
print("MAP L1")
print("Accuracy:",l1_accu)

