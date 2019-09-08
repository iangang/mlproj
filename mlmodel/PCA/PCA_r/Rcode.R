library(readr)
library(dplyr)
library(corrplot)
library(gplots)           # gplots是可视化包
library(RColorBrewer)
#-------------------------------------------------------------------------------
# 从网站下载该数据
airline <- read.csv("https://raw.githubusercontent.com/happyrabbit/DataScientistR/master/Data/AirlineRating.csv")
glimpse(airline)
#-------------------------------------------------------------------------------
# 用`corrplot()`函数检查问卷调查问题的相关性：
select(airline, Easy_Reservation:Recommend) %>%  # 选取其中的问卷调查项
    cor() %>%                                    # 得到相关矩阵
    corrplot(order = "hclust")                   # 用corrplot()绘制相关图
                                                 # 选项order="hclust"按照变量的相似度
                                                 # 基于系统聚类的结果对行列进行重新排列
#-------------------------------------------------------------------------------
# 主成分分析
airline.pc <- select(airline, Easy_Reservation:Recommend) %>%
    prcomp()
summary(airline.pc)

# 主成分分析陡坡图
plot(airline.pc, type="l", family ="Songti SC", main = "PCA陡坡图")

# PCA双标图
biplot(airline.pc, family ="Songti SC", 
       main = "PCA双标图", 
       cex = c(0.5, 1), 
       xlim = c(-0.06,0.04))
#-------------------------------------------------------------------------------
# 选取其中的问卷调查项和航空公司因子信息, 即删除ID项
airline.mean<-select(airline, -ID) %>%
    group_by(Airline) %>%               # 按Airline对数据进行分组总结
    summarise_each(funs(mean)) %>%      # 对每个数值
    glimpse()                           # 显示数据

# 聚合后PCA结果双标图
airline.mean.pc <- select(airline.mean, Easy_Reservation:Recommend) %>%
    prcomp()
biplot(airline.mean.pc, 
       family = "Songti SC", 
       main = "聚合后PCA结果双标图",
       cex = 0.7, 
       expand = 2, 
       xlim = c(-0.8, 1),
       ylim = c(-0.7,0.8))

#######-----------------------------------------------
# 将航空公司设置成行名称然后将对应的字符列删除
row.names(airline.mean) <- airline.mean$Airline
airline.mean <- select(airline.mean, -Airline)
# 绘制热图
heatmap.2(as.matrix(airline.mean),
          col = brewer.pal(9,"YlGn"),trace="none",key=FALSE,dend="none",cexCol=0.6,cexRow =1)
title(family ="Songti SC",
      main="航空公司问卷调查均值热图")

# 因子分析
library(GPArotation)
airline.fa<-airline%>%
    subset(select=Easy_Reservation:Recommend)%>%
    factanal(factors=3,rotation="oblimin")

library(gplots)
library(RColorBrewer)
# 绘制热图
heatmap.2(airline.fa$loadings,
          col=brewer.pal(9,"YlGn"),trace="none",key=FALSE,dend="none",cexCol=0.6,cexRow =1)
title(family ="Songti SC",
      main="航空公司满意度因子载荷")

# 因子得分
airline.fa <- airline %>%
    subset(select = Easy_Reservation:Recommend) %>%
    factanal(factors = 3, rotation = "oblimin", scores = "Bartlett")

fa.score <- airline.fa$scores %>%
    data.frame()rf

fa.score$Airline <- airline$Airline

fa.score.mean <- fa.score %>%
    group_by(Airline) %>%
    summarise(Factor1 = mean(Factor1),
              Factor2 = mean(Factor2),
              Factor3 = mean(Factor3))

row.names(fa.score.mean)<-as.character(fa.score.mean$Airline)
fa.score.mean<-select(fa.score.mean,-Airline)

heatmap.2(as.matrix(fa.score.mean),
          col=brewer.pal(9,"YlGn"),trace="none",key=FALSE,dend="none",cexCol=0.6,cexRow =1)
title(family ="Songti SC",
      main = "航空公司满意度平均因子分值")
