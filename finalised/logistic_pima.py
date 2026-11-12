import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay


df = pd.read_csv("../data/diabetes.csv")
x = df.drop("Outcome",axis=1)
y = df["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#without scaling
model = LogisticRegression(penalty=None,max_iter=1000)
model.fit(x_train,y_train)
pred = model.predict(x_test)
accu = accuracy_score(y_test,pred)
cm = confusion_matrix(y_test,pred)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['diabetes','no diabetes']).plot()
print("Without scaling.")
print("Accuracy:",accu)
plt.title("Without scaling")
plt.show()

#withScaling
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)

model.fit(x_train_scaled,y_train)
pred_scaled = model.predict(x_test_scaled)
accu_scaled = accuracy_score(y_test,pred_scaled)
cm_scaled = confusion_matrix(y_test,pred_scaled)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['diabetes','no diabetes']).plot()
print("With scaling.")
print("Accuracy:",accu_scaled)
plt.title("With scaled")
plt.show()

