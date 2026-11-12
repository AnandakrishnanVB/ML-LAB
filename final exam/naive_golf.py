import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

df = pd.read_csv("../data/play_golf.csv")

x = df.drop("PLAY GOLF",axis=1)
y = df["PLAY GOLF"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

encoder = OneHotEncoder()
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

multi = MultinomialNB()
multi.fit(x_train,y_train)
mulpred = multi.predict(x_test)
mulaccu = accuracy_score(y_test,mulpred)
mulf1 = f1_score(y_test,mulpred,pos_label='Yes')

ber = BernoulliNB()
ber.fit(x_train,y_train)
berpred = ber.predict(x_test)
beraccu = accuracy_score(y_test,berpred)
berf1 = f1_score(y_test,berpred,pos_label='Yes')

print("MultiNomial")
print("Accuracy:",mulaccu)
print("F1 score:",mulf1)

print("Bernoulli")
print("Accuracy:",beraccu)
print("F1 score:",berf1)
