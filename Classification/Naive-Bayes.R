# Naive Bayes 

# importing dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))

# splitting dataset into training and test sets
library(caTools)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling independent variables
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

# fitting naive bayes model to the training set
library(e1071)
classifier = naiveBayes(x = training_set[,1:2], y = training_set[,3])

# predicitng test set results
y_pred = predict(classifier, newdata = test_set)

# making confusion matrix
cm = table(test_set[,3], y_pred)

# visualizing training set results
library(ElemStatLearn)
X1 = seq(min(training_set[,1]) -1, max(training_set[,1] )+1, by =0.01)
X2 = seq(min(training_set[,2]) -1, max(training_set[,2] )+1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(training_set[,1:2], main = 'Naive Bayes - Training Set',
     xlim = range(X1), ylim = range(X2), xlab = 'Age', ylab = 'Salary')
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'black', 'grey'))
points(training_set, pch = 21, bg = ifelse(training_set[,3] == 1, 'yellow', 'pink'))

# visualising test set results
library(ElemStatLearn)
X1 = seq(min(test_set[,1]) -1, max(test_set[,1]) +1, by = 0.01)
X2 = seq(min(test_set[,2]) -1, max(test_set[,2]) +1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(test_set[,1:2], main = 'Naive Bayes - Test Set',
     xlim = range(X1), ylim = range(X2), xlab = 'Age', ylab = 'Salary')
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'orange', 'purple'))
points(test_set,pch =21, bg = ifelse(test_set[,3] == 1, 'white', 'green'))

