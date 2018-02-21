# Grid Search - Model's performance improvement
# Model Selection

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial',
                 C = 1, sigma = 2.25)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))

# applying grid search to find the optimal hyperparametres to improve model's performance
library(caret)
classifier = train(form = Purchased ~., data = training_set, method = 'svmRadial')
best_parameters = classifier$bestTune

# visualising training set results
library(ElemStatLearn)
X1 = seq(min(training_set[,1]) -1, max(training_set[,1]) +1,  by = 0.01)
X2 = seq(min(training_set[,2]) -1, max(training_set[,2]) +1,  by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(training_set[,1:2], main = 'Kernel SVM - Training Set',
     xlim = range(X1), ylim = range(X2), xlab = 'Age', ylab = 'Salary')
contour(X1,X2,matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(training_set, pch = 21, bg = ifelse(training_set[,3] == 1, 'yellow', 'blue'))

# visualising test set results
library(ElemStatLearn)
X1 = seq(min(test_set[,1] -1), max(test_set[,1] +1), by=0.01)
X2 = seq(min(test_set[,2]) -1, max(test_set[,2]) +1, by=0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(test_set[,1:2], main = 'Kernel SVM - Test Set',
     xlim = range(X1), ylim = range(X2), xlab = 'Age' , ylab = 'EstimatedSalary')
contour(X1,X2,matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'pink', 'brown'))
points(test_set, pch = 21, bg = ifelse(test_set[,3] == 1, 'purple', 'white'))
