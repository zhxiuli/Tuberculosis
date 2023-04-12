library(stringr)
library(scales)
library(magrittr)
library(RColorBrewer)
library(ggplot2)
library(pheatmap)
library(reshape2)
library(ComplexHeatmap)
library("ggsci")

df = read.csv('M_PC_data.csv',row.names = 1)
x = ggplot(df, aes(PC1,PC2,color = group)) + geom_point() +theme_bw() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") + 
  scale_color_npg() + theme(panel.grid = element_blank(),
                            legend.position = c(0.85,0.2)) + 
  labs(x = "PC1 (22.8%)",
       y = "PC2 (10.3%)")

df = read.csv('C_PC_data.csv',row.names = 1)
x = ggplot(df, aes(PC1,PC2,color = group)) + geom_point() +theme_bw() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") + 
  scale_color_npg() + theme(panel.grid = element_blank(),
                            legend.position = c(0.85,0.2)) + 
  labs(x = "PC1 (18.1%)",
       y = "PC2 (13.8%)")

df = read.csv('L_PC_data.csv',row.names = 1)
x = ggplot(df, aes(PC1,PC2,color = group)) + geom_point() +theme_bw() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") + 
  scale_color_npg() + theme(panel.grid = element_blank(),
                            legend.position = c(0.85,0.2)) + 
  labs(x = "PC1 (51.5%)",
       y = "PC2 (15.4%)")

df = read.csv('（蛋白）PC_data.csv',row.names = 1)
x = ggplot(df, aes(PC1,PC2,color = group)) + geom_point() +theme_bw() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") + 
  scale_color_npg() + theme(panel.grid = element_blank(),
                            legend.position = c(0.85,0.2)) + 
  labs(x = "PC1 (35.3%)",
       y = "PC2 (24.7%)")


df = read.csv('A_PC_data.csv',row.names = 1)
x = ggplot(df, aes(PC1,PC2,color = group)) + geom_point() +theme_bw() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") + 
  scale_color_npg() + theme(panel.grid = element_blank(),
                            legend.position = c(0.85,0.2)) + 
  labs(x = "PC1 (20.2%)",
       y = "PC2 (8.8%)")


