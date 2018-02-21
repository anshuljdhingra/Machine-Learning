# K-Means Clustering

# importing dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[,4:5]

# no splitting of data into training and test sets
# no feature scaling

# using the elbow method to find optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(dataset,i)$withinss)
plot(1:10, wcss, type = 'b', main = 'The Elbow Method', xlab = 'Number of Clusters', ylab = 'WCSS')

# fitting K-Means to the dataset
set.seed(99)
kmeans = kmeans(dataset, centers = 5, iter.max = 300, nstart = 10)
y_kmeans = kmeans$cluster

# visualing clusters
library(cluster)
clusplot(x = dataset, clus = y_kmeans, lines =2, shade = TRUE, color = TRUE, labels = 2, 
         plotchar = TRUE, span = TRUE, main = 'K-Means Clustering', xlab = 'Income', ylab = 'Spending')

