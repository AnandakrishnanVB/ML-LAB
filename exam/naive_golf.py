from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,f1_score
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
import numpy as np

outlook = [
    "rainy","sunny","overcast","overcast","sunny",
    "rainy","sunny","overcast","rainy","sunny",
    "sunny","rainy","overcast","overcast"
]

play = [
    "y","y","y","y","n","y","y",
    "y","n","n","y","n","y","y"
]
#We are using label encoder here because the inputs are categories.
#In the countVectorizer we use the words as input
#The 20NewsGroup uses text document.
encoder = LabelEncoder()
x = encoder.fit_transform(outlook).reshape(-1,1)
y = encoder.fit_transform(play)

#MultinomialNB
multi = MultinomialNB()
multi.fit(x,y)
multi_pred = multi.predict(x)

accuracy_multi = accuracy_score(y,multi_pred)
f1_multi = f1_score(y,multi_pred)

print("Multi accuracy:",accuracy_multi)
print("f1 score:",f1_multi)

print("-----------")


#Bernoullies
berno = BernoulliNB()
berno.fit(x,y)
berno_pred = berno.predict(x)

accuracy_berno = accuracy_score(y,berno_pred)
f1_berno = f1_score(y,berno_pred)

print("Berno accuracy:",accuracy_berno)
print("f1 score:",f1_berno)