# Linear Discriminant Analysis - Feature Extraction
# Dimensionality Reduction

# importing dataset
dataset = read.csv('Wine.csv')

# splitting the dataset into training and test sets
library(caTools)
set.seed(1234)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling
training_set[,1:13] = scale(training_set[,1:13])
test_set[,1:13] = scale(test_set[,1:13])

# applying lda
library(MASS)
lda_model = lda(formula = Customer_Segment ~., data = training_set)
training_set = as.data.frame(predict(lda_model, newdata = training_set))
test_set = as.data.frame(predict(lda_model, newdata = test_set))
training_set = training_set[c(5,6,1)]
test_set = test_set[c(5,6,1)]

# fitting svm to the training set
library(e1071)
classifier = svm(formula = class ~., 
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# predicitng test set results
y_pred = predict(classifier, newdata = test_set[,1:2])

# making confusion matrix
cm = table(test_set[,3], y_pred)

# visualising training set results
library(ElemStatLearn)
X1 = seq(min(training_set[,1])-1, max(training_set[,1])+1, by =0.01)
X2 = seq(min(training_set[,2]) -1, max(training_set[,2]) +1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(training_set[,1:2], main = 'LDA',
     xlim = range(X1), ylim = range(X2), xlab = 'LD1', ylab = 'LD2')
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse( y_grid ==2, 'white', ifelse(y_grid == 1, 'black', 'grey')))
points(training_set[,1:2], pch = 21, bg = ifelse(training_set[,3] == 2, 'red', 
                                                 ifelse(training_set[,3] == 1, 'yellow', 'blue')))


# visualising test set results
library(ElemStatLearn)
X1 = seq(min(test_set[,1]) -1, max(test_set[,1]) +1, by =0.01)
X2 = seq(min(test_set[,2]) -1, max(test_set[,2]) +1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('x.LD1','x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(test_set[,1:2],main = 'LDA', xlab = 'LD1', ylab = 'LD2',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse( y_grid ==2, 'orange', ifelse(y_grid == 1, 'green', 'pink')))
points(test_set[,1:2], pch = 21, bg = ifelse(test_set[,3] == 2, 'black', 
                                             ifelse(test_set[,3] == 1, 'yellow', 'blue')))
