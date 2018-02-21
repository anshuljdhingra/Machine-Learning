# Decision Tree Regression Model

# importing dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# splitting into test and training sets not required here because dataset is small
# feature scaling not required as Decision Tree Regression is not based on Euclidean Distance

#fitting Decision Tree Regression Model to dataset
#install.packages('rpart')
library(rpart)
set.seed(1234)

regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# predicting new values
y_pred = predict(regressor, data.frame(Level = 6.5))

# visualizing Decision Tree results
#library(ggplot2)
#ggplot() +
 # geom_point(aes(x = dataset$Level, y= dataset$Salary), color = 'red') +
#  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), color = 'blue') +
 # ggtitle('Level vs Salary') +
  #xlab('Level') +
  #ylab('Salary')

# visualising Decision Tree results for smooth curve and high resolution
library(ggplot2)

X_grid = seq(min(dataset$Level),max(dataset$Level),0.001)
ggplot() +
  geom_point(aes(x=dataset$Level,y=dataset$Salary), color='red')+
  geom_line(aes(x=X_grid,y=predict(regressor,newdata = data.frame(Level = X_grid))), color='blue')+
  ggtitle('Level vs Salary')+
  xlab('Level')+
  ylab('Salary')
