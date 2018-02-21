# Principal Component Analysis - Feature Extraction
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

# applying pca
library(caret)
library(e1071)
pca = preProcess(x=training_set[,1:13], method = 'pca', pcaComp = 2)
training_set = predict(pca, newdata = training_set)
test_set = predict(pca, newdata = test_set)
test_set = test_set[c(2,3,1)]
training_set = training_set[c(2,3,1)]

# fitting svm to the training set
library(e1071)
classifier = svm(formula = Customer_Segment ~., 
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# predicitng test set results
y_pred = predict(classifier, newdata = test_set[,1:2])

# making confusion matrix
cm = table(test_set[,3], y_pred)

# visualising training set results
library(ElemStatLearn)
X1 = seq(min(trainig_set[,1])-1, max(trainig_set[,1])+1, by =0.01)
X2 = seq(min(trainig_set[,2]) -1, max(trainig_set[,2]) +1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(training_set[,1:2], main = 'PCA',
     xlim = range(X1), ylim = range(X2), xlab = 'PC1', ylab = 'PC2')
contour(X1,X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse( y_grid ==2, 'orange', ifelse(y_grid == 1, 'green', 'pink')))
points(training_set[,1:2], pch = 21, bg = ifelse(training_set[,3] == 2, 'black', 
                                                ifelse(training_set[,3] == 1, 'yellow', 'blue')))


# visualising test set results
library(ElemStatLearn)
X1 = seq(min(test_set[,1]) -1, max(test_set[,1]) +1, by =0.01)
X2 = seq(min(test_set[,2]) -1, max(test_set[,2]) +1, by =0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('PC1','PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(test_set[,1:2],main = 'PCA', xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse( y_grid ==2, 'orange', ifelse(y_grid == 1, 'green', 'pink')))
points(test_set[,1:2], pch = 21, bg = ifelse(test_set[,3] == 2, 'black', 
                                                ifelse(test_set[,3] == 1, 'yellow', 'blue')))
