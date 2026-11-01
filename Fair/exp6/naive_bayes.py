
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

# Convert text into numbers
cv = CountVectorizer(stop_words='english')

train_x = cv.fit_transform(train.data)
test_x = cv.transform(test.data)

train_y = train.target
test_y = test.target

# Multinomial Naive Bayes
multi = MultinomialNB()
multi.fit(train_x, train_y)

pred1 = multi.predict(test_x)

acc1 = accuracy_score(test_y, pred1)
f1_1 = f1_score(test_y, pred1, average='weighted')

# Bernoulli Naive Bayes
ber = BernoulliNB()
ber.fit(train_x, train_y)

pred2 = ber.predict(test_x)

acc2 = accuracy_score(test_y, pred2)
f1_2 = f1_score(test_y, pred2, average='weighted')

# Display results
print("Multinomial Naive Bayes")
print("Accuracy:", acc1)
print("F1-score:", f1_1)

print("\nBernoulli Naive Bayes")
print("Accuracy:", acc2)
print("F1-score:", f1_2)
