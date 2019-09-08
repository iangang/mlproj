library(dendextend)

dend = c(1:5) %>% dist %>% hclust(method = "ave") %>% as.dendrogram
plot(dend)

labels(dend)

labels(dend) = c("A", "B", "extend", "dend", "C")
labels(dend)

labels_colors(dend)

labels_colors(dend) <- rainbow(5)
labels_colors(dend)
plot(dend)

cutree(dend, k = 2)
dend <- color_branches(dend, k = 2)
plot(dend)

dend2 <- sort(dend)
plot(dend2)

tanglegram(dend, dend2)

cor_cophenetic(dend, dend2 )
library(ggplot2)
ggplot(dend) 
