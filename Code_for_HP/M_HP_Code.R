# packages
library(stringr)
library(scales)
library(magrittr)
library(RColorBrewer)
library(ggplot2)
library(pheatmap)
library(reshape2)
library(ComplexHeatmap)
##############################################################
df = read.csv('M_HP.csv',row.names = 1)

x = pheatmap(df,scale = 'row',cellwidth = 15)
