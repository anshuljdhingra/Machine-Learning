# K-Fold Cross Validation - Model's Performance Evaulation
# Model Selection

# importing dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# spliting dataset into training and test sets
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling of independent variables
training_set[,1:2] =  scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

# fitting Kernel SVM model to training set
library(e1071)
classifier = svm(formula = Purchased ~. , data = training_set, 
                 type = 'C-classification', kernel = 'radial')

# predicting test set results
y_pred = predict(classifier, newdata = test_set[,1:2])

# making confusion matrix
cm = table(test_set[,3], y_pred)

# applyting k-fold to evaulate model's performance
library(caret)
folds = createFolds(training_set$Purchased, k =10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x,]
  test_fold = training_set[,x]
  classifier = svm(formula = Purchased ~., data = training_fold,
                   type = 'C-classification', kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[,-3])
  cm = table(test_fold[,3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) + (cm[1,2] + cm[2,1] + cm[1,1] + cm[2,2] )
  return (accuracy)
})
mean_accuracy = mean(as.numeric(cv))

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
