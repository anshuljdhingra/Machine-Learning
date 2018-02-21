# Eclat - Association Rule Learning

# importing dataset
library(arules)
dataset =  read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(x = dataset, topN = 15, popCol = 'black')

# fitting Eclat - Association Rule Learning Model to the dataset
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))
summary(rules)

# visualising model
inspect(sort(rules, by = 'support')[1:15])
