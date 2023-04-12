library(stringr)
library(scales)
library(magrittr)
library(RColorBrewer)
library(ggplot2)
library(pheatmap)
library(reshape2)
library(ComplexHeatmap)
##############################################################
df = read.csv('HP.csv',row.names = 1)
mat  = df
for (i in 1:nrow(mat)) mat[i, ] <- scale(log(unlist(mat[i, ]) + 1, 2))
mat <- as.matrix(mat)
mat[mat>2] = 2
mat[mat< -2] = -2
samples <- str_split(colnames(mat),'\\.',simplify = T)[,1]
gene.list <- c('Carnitine O-palmitoyltransferase II, mitochondrial variant (Fragment)',
               'Apolipoprotein E','Very-long-chain (3R)-3-hydroxyacyl-CoA dehydratase 2',
               'Selenoprotein P','Albumin (Fragment)','Extracellular matrix protein 1',
               'Insulin-like growth factor-binding protein 5','Inter-alpha (Globulin) inhibitor H2',
               'Inter-alpha-trypsin inhibitor heavy chain H1 (Fragment)','Inter-alpha (Globulin) inhibitor H2',
               'Leukocyte cell-derived chemotaxin-2','IgG L chain',
               "Inter-alpha (Globulin) inhibitor H2" )
`Lipid and amino acid metabolism` = c('Carnitine O-palmitoyltransferase II, mitochondrial variant (Fragment)',
                                      'Apolipoprotein E',
                                      'Albumin (Fragment)',
                                      'Insulin-like growth factor-binding protein 5',
                                      'Extracellular matrix protein 1',
                                                 'Very-long-chain (3R)-3-hydroxyacyl-CoA dehydratase 2','Selenoprotein P',
                                                 'Isocitrate dehydrogenase [NAD] subunit alpha, mitochondrial (Fragment)',
                                                 'Leucine-rich alpha-2-glycoprotein',
                                                 'Immunoglobulin alpha-2 heavy chain')

`Glucose metabolism` = c('Alpha-1-acid glycoprotein 2','Alpha-1-acid glycoprotein 1','Chromogranin-A',
                                    'Leukotriene A(4) hydrolase','Glycoprotein NMB','Insulin-like growth factor-binding protein 2',
                                    'Golgi membrane protein 1','Mannose receptor, C type 1-like 1','Neuropilin','Attractin',
                                    'Mast/stem cell growth factor receptor','Pigment epithelium-derived factor',
                                    'Plasma kallikrein','Insulin-like growth factor II transcript variant 3 isoform 1 (Fragment)',
                                    'Cadherin-13','Rho GDP-dissociation inhibitor 2 (Fragment)','Fibronectin 1 (FN1),transcript variant 4')
`Inflammatory response` = c('Alpha-1-antitrypsin','C-reactive protein','Chitotriosidase-1','Complement C3 (Fragment)',
                            'Complement component 9, isoform CRA_a','Inter-alpha-trypsin inhibitor heavy chain H3',
                            'Inter-alpha-trypsin inhibitor heavy chain H4','Leukocyte cell-derived chemotaxin-2',
                            'Serpin A11','Cathepsin Z','Metalloproteinase inhibitor 1','Intercellular adhesion molecule 1',
                            'Cathepsin D','von Willebrand factor','Complement C4A3 (Fragment)',
                            'Inter-alpha (Globulin) inhibitor H2',
                            'Inter-alpha (Globulin) inhibitor H2',
                            'D-dopachrome decarboxylase')
gene.group = as.data.frame(rownames(mat))
gene.group$group = 'Others'
gene.group$group = ifelse(gene.group$`rownames(mat)` %in% `Glucose metabolism`,'Glucose metabolism',gene.group$group)
gene.group$group = ifelse(gene.group$`rownames(mat)` %in% `Lipid and amino acid metabolism`,'Lipid and amino acid metabolism',gene.group$group)
gene.group$group = ifelse(gene.group$`rownames(mat)` %in% `Inflammatory response`,'Inflammatory response',gene.group$group)

x = Heatmap(mat[which(rownames(mat) %in% gene.group[which(gene.group$group == "Glucose metabolism"),'rownames(mat)'] ),], 
        col = colorRampPalette(rev(brewer.pal(n = 7, name ="RdYlBu")))(100), 
        heatmap_legend_param = list(grid_height = unit(10,'mm')),  
        show_row_names = T,  
        width = ncol(mat)*unit(5, "mm"), 
        height = ncol(mat)*unit(4, "mm"), 
        column_split = samples,
        # row_split = factor(gene.group$group),
        top_annotation = HeatmapAnnotation(Group = samples,
                                           simple_anno_size = unit(2, 'mm'), 
                                           col = list(Group = c('con' = '#E77273', 'neg' = '#719AE1', 'pos' = '#26A99E')),  
                                           show_annotation_name = FALSE), 
        column_names_gp = gpar(fontsize = 10), row_names_gp = gpar(fontsize = 6))

x = Heatmap(mat[which(rownames(mat) %in% gene.group[which(gene.group$group == "Inflammatory response"),'rownames(mat)'] ),], 
            col = colorRampPalette(rev(brewer.pal(n = 7, name ="RdYlBu")))(100), 
            heatmap_legend_param = list(grid_height = unit(10,'mm')),  
            show_row_names = T,  
            width = ncol(mat)*unit(5, "mm"), 
            height = ncol(mat)*unit(4, "mm"), 
            column_split = samples,
            # row_split = factor(gene.group$group),
            top_annotation = HeatmapAnnotation(Group = samples,
                                               simple_anno_size = unit(2, 'mm'), 
                                               col = list(Group = c('con' = '#E77273', 'neg' = '#719AE1', 'pos' = '#26A99E')),  
                                               show_annotation_name = FALSE), 
            column_names_gp = gpar(fontsize = 10), row_names_gp = gpar(fontsize = 6))

x = Heatmap(mat[which(rownames(mat) %in% gene.group[which(gene.group$group == "Lipid and amino acid metabolism"),'rownames(mat)'] ),], 
            col = colorRampPalette(rev(brewer.pal(n = 7, name ="RdYlBu")))(100), 
            heatmap_legend_param = list(grid_height = unit(10,'mm')),  
            show_row_names = T,  
            width = ncol(mat)*unit(5, "mm"), 
            height = ncol(mat)*unit(4, "mm"), 
            column_split = samples,
            # row_split = factor(gene.group$group),
            top_annotation = HeatmapAnnotation(Group = samples,
                                               simple_anno_size = unit(2, 'mm'), 
                                               col = list(Group = c('con' = '#E77273', 'neg' = '#719AE1', 'pos' = '#26A99E')),  
                                               show_annotation_name = FALSE), 
            column_names_gp = gpar(fontsize = 10), row_names_gp = gpar(fontsize = 6))
x = Heatmap(mat, 
            col = colorRampPalette(rev(brewer.pal(n = 7, name ="RdYlBu")))(100), 
            heatmap_legend_param = list(grid_height = unit(10,'mm')),  
            show_row_names = T,  
            width = ncol(mat)*unit(5, "mm"), 
            column_split = samples,
            # row_split = factor(gene.group$group),
            top_annotation = HeatmapAnnotation(Group = samples,
                                               simple_anno_size = unit(2, 'mm'), 
                                               col = list(Group = c('con' = '#E77273', 'neg' = '#719AE1', 'pos' = '#26A99E')),  
                                               show_annotation_name = FALSE), 
            column_names_gp = gpar(fontsize = 10), row_names_gp = gpar(fontsize = 6))

##############################################################
library(ggpubr)
df = mat[which(rownames(mat) %in% gene.group[which(gene.group$group == "Inflammatory response"),'rownames(mat)'] ),]

colnames(df) = c('Protein','Group','Expression')
df.mean <- aggregate(Expression ~ Group+Protein, df, mean)
df.mean$sd <- aggregate(Expression ~ Group+Protein, df, sd)$Expression
df.mean$se <- df.mean$sd / sqrt(3)

x = ggplot(df.mean, aes(x=Group, y=Expression, group=1)) +
  geom_line(aes(color=Group), size=1.2) +
  geom_point(aes(color=Group), size=3) +
  geom_errorbar(aes(ymin=Expression-se, ymax=Expression+se), width=.2) +
  scale_color_manual(values=c("#1F77B4", "#FF7F0E", "#2CA02C")) +
  theme_classic() + facet_wrap(~Protein,nrow = 4) + 
  theme(legend.position = c(0.75,0.2))+ stat_compare_means()
x

library(stats)
a = unique(df$Protein)
for(i in a){
  df.ids = df[which(df$Protein == i),]
  p.val = t.test(df.ids$Expression[df.ids$Group == "neg"], df.ids$Expression[df.ids$Group == "con"])
  print(paste0(i,': ',as.character(format(p.val$p.value, digits=3))))
}
for(i in a){
  df.ids = df[which(df$Protein == i),]
  p.val = t.test(df.ids$Expression[df.ids$Group == "pos"], df.ids$Expression[df.ids$Group == "con"])
  print(paste0(i,': ',as.character(format(p.val$p.value, digits=3))))
}









