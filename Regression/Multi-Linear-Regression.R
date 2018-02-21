# Mutlti Linear Regression

# Importing Dataset
dataset = read.csv('50_Startups.csv')

# Encoding Categorical Values

dataset$State = factor(x=dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))

# splitting dataset into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# fitting linear regression model for training set
regressor = lm( formula = Profit ~ ., data = training_set)
summary(regressor)

# predicting test set results
y_pred = predict(regressor, newdata = test_set )

#  building the optimal model using backward elimintion method
regressor = lm( formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = training_set)
summary(regressor)

regressor = lm( formula = Profit ~ R.D.Spend ,
                data = training_set)
summary(regressor)
