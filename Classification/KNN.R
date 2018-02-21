# K-Nearest Neighbors Classification

# importing dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# splitting dataset into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling for the independent variables
training_set[,-3] = scale(training_set[,-3])
test_set[,-3] = scale(test_set[-3])

# fitting the KNN classifier and predicting test results
library(class)
y_pred = knn(train = training_set[,-3], test = test_set[,-3],cl = training_set[,3],k=5, prob=TRUE)

# making confusion matrix
cm = table(test_set[,3],y_pred)

# visualising training set results
library(ElemStatLearn)
set =  training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = training_set[,-3],test = grid_set, cl = training_set[,3], k=5, prob = TRUE)
plot(training_set[,-3], main = 'K-NN Classification - Training Set', 
     xlab = 'age', ylab = 'salary', xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(training_set[, -3], pch = 21, bg = ifelse(training_set[,3]==1, 'white', 'black'))


# visualizing test set results
library(ElemStatLearn)
X1 = seq(min(test_set[1]) -1, max(test_set[1]) + 1, by = 0.01)
X2 = seq(min(test_set[2]) -1, max(test_set[2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
grid_pred = knn(train = test_set[, 1:2], test = grid_set, cl = test_set[,3],k=5, prob = TRUE)
plot(test_set[,1:2], main = 'K-NN Classfication - Test Set', xlab = 'Age', ylab = 'Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(grid_pred), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(grid_pred ==1, 'springgreen3', 'tomato'))
points(test_set[,1:2], pch = 21, bg = ifelse(y_pred == 1, 'blue', 'white'))
