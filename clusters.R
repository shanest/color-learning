library(mclust)
data <- read.table('trial/results.csv', sep=',', header=TRUE)
# data <- data[c('degree_of_convexity', 'accuracy')]
D = Mclust(data[c('degree_of_convexity', 'accuracy')],G=1:20)
summary(D)
plot(D, what="classification")

BIC <- mclustBIC(data)
plot(BIC)
summary(BIC)

library(ggplot2)

# qplot(degree_of_convexity, accuracy, data=data, geom=c('point', 'smooth'), method="lm", formula=y~x)
data$cluster <- D['classification']
data$cluster <- as.factor(data$cluster)
data$temp <- as.factor(data$temp)
data$conv <- as.factor(data$conv)
ggplot(full_data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=conv))