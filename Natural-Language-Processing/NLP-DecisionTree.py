# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:44:49 2017

@author: adhingra
"""
# Natural Lanugage Processing
# using Decision Tree

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ' , dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Fitting Bag of words model to the corpus
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1400)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting dataset into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Decision Tree Classification Model to the dataset
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

# predicting test set results
y_pred = classifier.predict(X_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# getting accuracy, preicison, recall , f1 score and classification report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
classificationreport = classification_report(y_test,y_pred)
