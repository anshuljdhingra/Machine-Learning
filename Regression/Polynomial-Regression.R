# Polynomial Linear Regression

# importing dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
#dataset1 = dataset[,2:3]

# fitting linear regression model to the dataset
lin_reg = lm(formula = Salary ~ .,data = dataset)

# fitting polynomial regression model to dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset)

# visualising linear regression model
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), color = 'blue') +
  ggtitle('Level vs Salary') +
  xlab('Level') +
  ylab('Salary')

# visualising polynomial regression model
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), color = 'blue')+
  ggtitle('Level vs Salary') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid,
                                                                              Level2 = x_grid^2,
                                                                              Level3 = x_grid^3,
                                                                              Level4 = x_grid^4))),
                color = 'blue')+
  ggtitle('Level vs Salary') +
  xlab('Level') +
  ylab('Salary')

# predicting values using linear regression
y_pred_linear = predict(lin_reg, newdata = data.frame(Level = 6.5))

# predicting values using polynomial regression
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                     Level2 = 6.5^2,
                                                     Level3 = 6.5^3,
                                                     Level4 = 6.5^4))
