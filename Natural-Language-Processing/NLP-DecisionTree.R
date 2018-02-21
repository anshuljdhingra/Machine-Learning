# Natural Language Processing 
# using Decision Tree Classification Model

# importing dataset
dataset = read.delim('Restaurant_Reviews.tsv', sep = '\t', quote = '', stringsAsFactors = FALSE)

# cleaning the texts
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# fitting bag of words model
dtm = DocumentTermMatrix(corpus)
dataset_dtm = removeSparseTerms(dtm, 0.999)
dataset_dtm = as.data.frame(as.matrix(dataset_dtm))
dataset_dtm$Liked = dataset$Liked

# encoding the dependent variable as factor
dataset_dtm$Liked = factor(x = dataset_dtm$Liked, levels = c(0,1))

# splitting the dataset into training and test sets
library(caTools)
split = sample.split(dataset_dtm$Liked , SplitRatio = 0.8)
training_set = subset(dataset_dtm, split == TRUE)
test_set = subset(dataset_dtm, split == FALSE)

# fitting decision tree classification model to the new dataset
library(rpart)
classifier = rpart(formula = Liked ~ ., data = training_set)

# predicting test set results
y_pred = predict(classifier, newdata = test_set[,1:691], type = 'class')

# making confusion matrix
cm = table(test_set[,692],y_pred)

# getting accuracy, precision, f1 and recall score
library(caret)
accuracy = (cm[4] + cm[1]) / (cm[1] + cm[2] + cm[3] + cm[4])
precision_score = cm[4] / (cm[4] + cm[3])
recall_score = cm[4] / (cm[4] + cm[2])
f1 = (2*precision_score*recall_score)/(precision_score+recall_score)
