import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,f1_score

df = pd.read_csv("../data/play_golf.csv")

x = df.drop("PLAY GOLF",axis=1)
y = df["PLAY GOLF"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

encoder = OneHotEncoder(handle_unknown="ignore")
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

#Multinomial
multi = MultinomialNB()
multi.fit(x_train,y_train)
multi_pred = multi.predict(x_test)

multi_accuracy = accuracy_score(y_test,multi_pred)
multi_f1 = f1_score(y_test,multi_pred,pos_label="Yes")
print("----MultiNomial----")
print("Accuracy:",multi_accuracy)
print("F1 score:",multi_f1)

#Bernoulli
berno = BernoulliNB()
berno.fit(x_train,y_train)
berno_pred = berno.predict(x_test)

berno_accuracy = accuracy_score(y_test,berno_pred)
berno_f1 = f1_score(y_test,berno_pred,pos_label="Yes")

print("----Bernoulli----")
print("Accuracy:",berno_accuracy)
print("f1 score:",berno_f1)