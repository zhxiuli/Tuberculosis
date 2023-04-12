### ---------------
### Create: Yuan.Sh, MD (ORCID: 0000-0002-6028-0185)
### Date: Feb. 28, 2023
### Email: yuansh3354@163.com
### Blog: https://blog.csdn.net/qq_40966210
### Github: https://github.com/yuansh3354/
### Official Account: DeepBioinformatics
### Address:
###         1. Fujian Medical University. No. 1 Xue Yuan Road,
###            University Town, 350122 FuZhou Fujian, China.
###         2. National Center for Nanoscience and Technology (NCNST).
###            No.11 ZhongGuanCun BeiYiTiao, 100190 Beijing, China.
### ---------------
########################## Step.00 ########################## 
# clean
rm(list = ls())
gc()

# packages
library(stringr)
library(scales)
library(magrittr)
library(RColorBrewer)
library(ggplot2)

# myfunctions 
head1 = function(x,DT=FALSE){
  print(paste('Data dimension: ',dim(x)[1],dim(x)[2]))
  if(dim(x)[2]>5){print(x[1:5,1:5])}
  else(print(head(x)))
  if(DT){DT::datatable(x,filter='top')}
}

mycolors = c("#1a476f","#90353b","#55752f","#e37e00","#6e8e84",
             "#c10534","#938dd2","#cac27e","#a0522d","#7b92a8",
             "#2d6d66","#9c8847","#bfa19c","#ffd200","#d9e6eb",
             "#4DBBD5B2", "#00A087B2", "#3C5488B2", 
             "#F39B7FB2", "#8491B4B2","#91D1C2B2","#DC0000B2",
             "#7E6148B2","#E64B35B2",'#698EC3')
show_col(mycolors)

# options
options(stringsAsFactors = F)
options(as.is = T)
##############################################################
setwd("/Volumes/lexar/巨噬细胞-菌阴结核识别资料/3-新疆菌阴结核病多组学相关/2018蛋白报告及附件/报告及附件/质谱鉴定和定量结果")

df1 = read.table('negvscon.txt',fill = T,header = T)






