# Decision Tree Classification (using feature scaling for high resolution graphs)

# importing dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))

# splitting dataset into training and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling of the independent variables
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

# fitting Decision Tree Classification Model to the training set
library(rpart)
classifier = rpart(formula = Purchased ~., data = training_set)

# predicting test set results
y_pred = predict(classifier, newdata = test_set[,1:2], type = 'class')

# making confusion matrix
cm = table(test_set[,3], y_pred)

# visualising training set results
library(ElemStatLearn)
X1 = seq(min(training_set[,1]) -1, max(training_set[,1]) +1, by = 0.01)
X2 = seq(min(training_set[,2]) -1, max(training_set[,2]) +1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(training_set[,1:2], main = 'Decision Tree - Training Set',
     xlim = range(X1), ylim = range(X2), xlab = 'Age', ylab = 'Salary')
contour(X1,X2, matrix(as.numeric(y_grid) , length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'white', 'black'))
points(training_set, pch = 21, bg = ifelse(training_set[,3] == 1, 'blue', 'yellow'))

# visualising test set results
library(ElemStatLearn)
X1 = seq(min(test_set[,1]) -1, max(test_set[,1]) +1, by = 0.01)
X2 = seq(min(test_set[,2]) -1, max(test_set[,2]) +1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(training_set[,1:2], main = 'Decision Tree - Test Set',
     xlim = range(X1), ylim = range(X2), xlab = 'Age', ylab = 'Salary')
contour(X1,X2, matrix(as.numeric(y_grid) , length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'orange', 'violet'))
points(test_set, pch = 21, bg = ifelse(test_set[,3] == 1, 'blue', 'yellow'))

# making decision tree
plot(classifier)
text(classifier)
