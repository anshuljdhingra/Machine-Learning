# Apriori - Association Rule Learning

# importing dataset
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 50)

# Fitting Apriori - Association Rule Learning Model to the dataset
#rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.8))

#rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.4))

#rules = apriori( data = dataset, parameter = list(support = 0.003, confidence = 0.2))

rules = apriori( data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# visualising apriori - association rules
inspect(sort(rules , by = 'lift')[1:10])
