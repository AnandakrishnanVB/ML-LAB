from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,f1_score

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

cv = CountVectorizer(stop_words='english')
x_train = cv.fit_transform(train.data)
x_test = cv.transform(test.data)

y_train = train.target
y_test = test.target

multi = MultinomialNB()
multi.fit(x_train,y_train)
mulpred = multi.predict(x_test)
mulaccu = accuracy_score(y_test,mulpred)
mulf1 = f1_score(y_test,mulpred,average='weighted')

ber = BernoulliNB()
ber.fit(x_train,y_train)
berpred = ber.predict(x_test)
beraccu = accuracy_score(y_test,berpred)
berf1 = f1_score(y_test,berpred,average='weighted')


print("Mulitnomial")
print("Accuracy:",mulaccu)
print("F1 score:",mulf1)

print("BernoulliNB")
print("Accuracy:",beraccu)
print("F1 score:",berf1)