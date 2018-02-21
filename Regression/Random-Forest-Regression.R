# Random Forest Regression

# importing dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# splitting into training and test set not reqd as dataset is smaller
# feature scaling not reqd as this is not based on Euclidean Distance

# fiting dataset into Random Forest Regerssion Model
#install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x=dataset[1],
                         y=dataset$Salary,
                         ntree = 500)

# predicting new results
y_pred = predict(regressor,newdata = data.frame(Level = 6.5))

# visualising Random Forest Regression results (for high resolution and smooth curve)
library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level),0.01)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),color='red')+
  geom_line(aes(x=x_grid,y=predict(regressor,newdata = data.frame(Level = x_grid))), color='blue')+
  ggtitle('Salary vs Level')+
  xlab('Level')+
  ylab('Salary')



