# Hierarchical Clustering 

# importing dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[,4:5]

# using dendrograms to find the optimal number of clusters
dendrogram = hclust(dist(dataset, method = 'euclidean'), method = 'ward.D')
plot(dendrogram, main = paste('Dendrogram'), xlab = 'Customers', ylab = 'Euclidean Distance')

# fitting Hierarchical Clustering to the dataset
hc = hclust(dist(dataset, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# visualing the HC
library(cluster)
clusplot( dataset,y_hc, lines = 0, shade = TRUE, color = TRUE, labels = 4, plotchar = FALSE,
          span = TRUE, main = paste('Hierarchical Clustering of Customers'), xlab = 'Income', ylab = 'Spending')
