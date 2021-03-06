# Logistic Regression

# importing dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# splitting data into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature scaling for X
training_set[-3]= scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# fitting logistic regression model to training set
classifier = glm(formula = Purchased ~ ., family = binomial, data= training_set)

# predicting test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred >0.5, 1, 0)

# making confusion matrix
cm = table(test_set[,3],y_pred)

# visualizing training set results 
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[1])-1,max(set[1])+1,by=0.01)
X2 = seq(min(set[2])-1, max(set[2])+1,by=0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier,type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5,1,0)
plot(set[,-3], main = 'logistic regression - training set', xlab = 'age', ylab = 'salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid)), length(X1), length(X2),add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid ==1, 'springgreen', 'tomato'))
points(set , pch = 21, bg = ifelse(set[,3]==1, 'white', 'black'))

# visualizing test set results
library(ElemStatLearn)
set=test_set
X1 = seq(min(set[1])-1,max(set[1])+1, by=0.01)
X2 = seq(min(set[2])-1, max(set[2])+1, by=0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_pred = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_pred > 0.5, 1, 0)
plot(set[, -3], main = 'logistic regression - test result', xlab = 'age', ylab = 'salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid)), length(X1), length(X2), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==1, 'springgreen', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3]==1, 'yellow', 'blue'))
         
