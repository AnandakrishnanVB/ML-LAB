#Naïve Bayes is a supervised machine learning algorithm used for classification.
#It uses Bayes' theorem and assumes that features are conditionally independent given the class.

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,f1_score
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
#Multinomial → counts how many times a word appears.
#Bernoulli → checks whether a word appears or not.

#the dataset comes as the train and test data so dont need to split.
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

#The count vectorizer is used to convert the words into numerical word counts.
#stop_words="english" -> this will removes common English words like "the","is"..
cv = CountVectorizer(stop_words="english")
x_train = cv.fit_transform(train.data)
x_test = cv.transform(test.data)

#The targer values already given as numbers.
y_train = train.target
y_test = test.target

#Multinomial
multi = MultinomialNB()
multi.fit(x_train,y_train)
multi_pred = multi.predict(x_test)

accuracy_multi = accuracy_score(y_test,multi_pred)
f1_multi = f1_score(y_test,multi_pred,average='weighted')

#bernouli
berno = BernoulliNB()
berno.fit(x_train,y_train)
berno_pred = berno.predict(x_test)

accuracy_berno = accuracy_score(y_test,berno_pred)
f1_berno = f1_score(y_test,berno_pred,average="weighted")

print("----Bernoully----")
print("Accuracy:",accuracy_berno)
print("f1 score:",f1_berno)

print("----Mulitnomial----")
print("Accuracy:",accuracy_multi)
print("f1 score:",f1_multi)