# Random Forest Classification (using feature scaling for high resolution graphs)

# importing dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# encoding dependent variable as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))
                           
# splitting data into training and test sets
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling of independent variables
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

# fitting Random Forest Classification to the training set
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[,1:2], y=training_set[,3], ntree = 100)

# prediciting test set results
y_pred = predict(classifier, newdata = test_set[,1:2])
      
# making confusion matrix
cm = table(test_set[,3], y_pred)

# visualising training set results
library(ElemStatLearn)
X1 = seq(min(training_set[,1]) -1, max(training_set[,1]) +1 , by =0.01)
X2 = seq(min(training_set[,2]) -1, max(training_set[,2]) +1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(training_set[,1:2], main = 'Random Forest - Training Set', 
     xlim = range(X1), ylim = range(X2), xlab = 'Age', ylab = 'Salary')
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'cyan3', 'darkgoldenrod2'))
points(training_set, pch =21, bg = ifelse(training_set[,3] == 1, 'coral', 'burlywood'))

# visualising test set results
library(ElemStatLearn)
X1 = seq(min(test_set[,1]) -1, max(test_set[,1]) +1 , by =0.01)
X2 = seq(min(test_set[,2]) -1, max(test_set[,2]) +1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(test_set[,1:2], main = 'Random Forest - Test Set', 
     xlim = range(X1), ylim = range(X2), xlab = 'Age', ylab = 'Salary')
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'brown2', 'cornsilk'))
points(test_set, pch =21, bg = ifelse(test_set[,3] == 1, 'cyan', 'chartreuse'))

# plotting Decision Tree
plot(classifier)
