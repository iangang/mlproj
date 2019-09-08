# 聚类可视化方法


## 散点图可视化

```r
res = kmeans(df, cneters = 3)
pairs(df, col = res$cluster +１)
```

```r
if(!require(GGally)) install.packages("GGally")

ggpairs(df, columns = 1:5, mapping = aes(colour = as.character(res$cluster)))
```


## 段剖面图

> 高维数据聚类结果可视化；

```r
barchart(res, legend = TRUE)
```