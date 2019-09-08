# install.packages("d3heatmap")
library(d3heatmap)

url <- "http://datasets.flowingdata.com/ppg2008.csv"
nba_players <- read.csv(url, row.names = 1)
head(nba_players)
d3heatmap(nba_players, scale = "column")
d3heatmap(nba_players, scale = "column", dendrogram = "none", color = "Blues")
d3heatmap(nba_players, scale = "column", dendrogram = "none", color = scales::col_quantile("Blues", NULL, 5))
d3heatmap(nba_players, colors = "Blues", scale = "col", endrogram = "row", k_row = 3)
