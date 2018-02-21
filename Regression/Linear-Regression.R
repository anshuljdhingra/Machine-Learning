# Simple Linear Regression

# importing dataset
dataset = read.csv('Salary_Data.csv')

# Splitting dataset into training and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Simple Linear Regression to the Training Set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# predicting test results
y_pred = predict(regressor, newdata = test_set)

# visualising training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y= training_set$Salary), colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set),
                 colour = 'black')) +
  ggtitle('Salary vs Exp [Training Set]') +
  xlab('Exp') +
  ylab('Salary')

# visualizing test set results
library(ggplot2)
ggplot() +
  geom_point(aes(x=test_set$YearsExperience,y=test_set$Salary), color = 'red') +
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata = training_set)), color = 'blue') +
  ggtitle('Exp vs Salary [Test Set]') +
  xlab('Exp') +
  ylab('Salary')
