#Naïve Bayes is a supervised machine learning algorithm used for classification.
#It uses Bayes' theorem and assumes that features are conditionally independent given the class.

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.navie_bayes import BernoulliNB,MultinomialNB
from sklearn.metrics import accuracy_score,f1_score


train = fetch_20newsgroups(subset='train')
text = fetch_20newsgroups(subset='test')

cv = CountVectorizer(stop_words="english")
