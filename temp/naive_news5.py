from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB,BernoulliNB


train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

cv = CountVectorizer(stop_words='english')
x_train = cv.fit_transform(train.data)
x_test = cv.transform(test.data)

y_train = train.target
y_test = test.target


#multi
multi = MultinomialNB()
multi.fit(x_train,y_train)
multi_pred = multi.predict(x_test)

multi_accuracy = accuracy_score(y_test,multi_pred)
print("Multi")
print("Accuracy:",multi_accuracy)

#berboulis