library(mclust)
data <- read.table('data/results.csv', sep=',', header=TRUE)

data$min_size <- apply(
  data[grepl('size', colnames(data))], 1, min
)
data$max_size <- apply(
  data[grepl('size', colnames(data))], 1, max
)
data$max_over_min <- data$max_size / data$min_size
data$median_size <- apply(
  data[grepl('size', colnames(data))], 1, median
)
data$mean_size <- apply(
  data[grepl('size', colnames(data))], 1, mean
)
# data <- data[c('degree_of_convexity', 'accuracy')]
D = Mclust(data[c('degree_of_convexity', 'accuracy')],G=1:20)
summary(D)
png('clusters.png', width=12, height=9, units='in', res=300)
plot(D, what="classification")
dev.off()

BIC <- mclustBIC(data)
png('cluster_BIC.png', width=12, height=9, units='in', res=300)
plot(BIC)
dev.off()
summary(BIC)


library(ggplot2)
library(viridis)

# qplot(degree_of_convexity, accuracy, data=data, geom=c('point', 'smooth'), method="lm", formula=y~x)
data$cluster <- as.factor(D$classification)
data$temp <- as.factor(data$temp)
data$conv <- as.factor(data$conv)
ggplot(data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=conv))
ggplot(data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=temp))
ggplot(data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=linear_accuracy)) + scale_colour_viridis()
ggplot(data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=min_size)) + scale_colour_viridis()
ggplot(data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=max_size)) + scale_colour_viridis()
ggplot(data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=max_over_min)) + scale_colour_viridis()
ggplot(data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=median_size)) + scale_colour_viridis()
ggsave('clusters_median.png', units='in', width=18, height=12)
ggplot(data) + geom_point(aes(x=degree_of_convexity, y=accuracy, shape=cluster, colour=mean_size)) + scale_colour_viridis()

no_bad_cluster <- data[which(data$cluster != 3), ]
regress <- lm(accuracy ~ degree_of_convexity, data=no_bad_cluster)
summary(regress)
