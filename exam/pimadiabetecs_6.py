import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,ConfusionMatrixDisplay

df = pd.read_csv("../data/diabetes.csv")

x = df.drop(columns=['Outcome']) #We can also do dr.drop("Outcome",axis=1) for drop the column.
#axis = 0 -> row
#axis = 1 -> column
y = df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

model = LogisticRegression(penalty=None,max_iter=1000)

#Without feature scaling
model.fit(x_train,y_train)
pred = model.predict(x_test)

precision = precision_score(y_test,pred)
accuracy = accuracy_score(y_test,pred)
f1 = f1_score(y_test,pred)
recall = recall_score(y_test,pred)
cm = confusion_matrix(y_test,pred)

print("----Without feature scaling----")
print("Precision:",precision)
print("Accuracy:",accuracy)
print("F1 score:",f1)
print("Recall:",recall)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["no diabetes","diabetes"]).plot()
plt.title("Without feature scaling")
plt.show()


#with feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model.fit(x_train_scaled,y_train)
scaled_pred = model.predict(x_test_scaled)

scaled_precision = precision_score(y_test,scaled_pred)
scaled_accuracy = accuracy_score(y_test,scaled_pred)
scaled_f1 = f1_score(y_test,scaled_pred)
scaled_recall = recall_score(y_test,scaled_pred)
scaled_cm = confusion_matrix(y_test,scaled_pred)

print("----With feature scaling----")
print("Precision:",scaled_precision)
print("Accuracy:",scaled_accuracy)
print("F1 score:",scaled_f1)
print("Recall:",scaled_recall)
ConfusionMatrixDisplay(confusion_matrix=scaled_cm,display_labels=["no diabetes","diabetes"]).plot()
plt.title("With feature scaling")
plt.show()
