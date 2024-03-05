library(openxlsx)
library(ggplot2)
library(stringr)
library(enrichplot)
library(clusterProfiler)

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
#BiocManager::install('Installation package name')


#Read differentially expressed genes and convert gene IDs from GENE_SYMBOL to ENTREZ_ID:
info <- read.xlsx( "./gene.xlsx", rowNames = F,colNames = T)
chr_num <- c()
for (i in 1:length(info$snp)) {
  chr_num[i] = strsplit(info$snp[i],":")[[1]][1]
  
}

info$snp_chr_num <- chr_num

#Selection of chromosome sets for analysis
library(stats)
cls_attn <- c(0.709489281, 0.756021438, 0.713430013, 0.71519546, 0.714754098, 0.70204918, 0.715384615, 0.67203657, 0.7360971, 0.657739596, 0.723581337, 0.710687264, 0.629602774, 0.69908575, 0.674779319, 0.654066835, 0.698738966, 0.688839849, 0.728972257, 0.702112232, 0.601481715, 0.678846154)
#cls_attn <- c(0.781455083, 0.699130357, 0.703760983, 0.659606419, 0.722770979, 0.690064103, 0.744220683, 0.684081944, 0.698942084, 0.743587502, 0.741314775, 0.765711852, 0.674465438, 0.726526358, 0.665300565, 0.757555585, 0.672581585, 0.714532903, 0.713923256, 0.724042944, 0.674648108, 0.689453335)
mean_cls_attn <- mean(cls_attn)
sd_cls_attn <- sd(cls_attn)
z_scores <- qnorm(p = 0.5, mean = mean_cls_attn, sd = sd_cls_attn)
significant_positions <- which(cls_attn > z_scores)
significant_num <- length(significant_positions)
print(paste("Notable in total are：",significant_num))

index <- info$snp_chr_num %in% significant_positions
gene_list <- info$gene[index]
gene_list <- str_trim(gene_list)
gene_list <- gene_list[gene_list != "NULL"]
gene_list <- unique(gene_list)

## Specify the species library for enrichment analysis
KEGG_database <- 'hsa' #KEGG analysis of specified species, species abbreviation index table detailed at http://www.genome.jp/kegg/catalog/org_list.html

#gene ID conversion
gene <- bitr(gene_list,fromType = 'SYMBOL',toType = 'ENTREZID',OrgDb = 'org.Hs.eg.db')
failed_genes <- gene_list[is.na(gene$ENTREZID)]

KEGG<-enrichKEGG(gene$ENTREZID,
                 organism = KEGG_database,
                 pvalueCutoff = 0.05,
                 qvalueCutoff = 0.05) 


#Drawing Enrichment Pathways

barplot(KEGG,showCategory = 40,title = 'KEGG Pathway')
dotplot(KEGG)
KEGG$p.adjust
KEGG$Description
KEGG$ID

#Association network diagram of enriched genes with the set of functions/pathways in which they are located
enrichplot::cnetplot(KEGG,circular=FALSE)
enrichplot::heatplot(KEGG,showCategory = 50)

#Correlation network diagram between enriched function sets/pathway sets
KEGG2 <- pairwise_termsim(KEGG)
enrichplot::emapplot(KEGG2,showCategory =50, color = "p.adjust")

#Preservation of Pathways
write.table(KEGG$ID, file = "./KEGG_IDs.txt", 
            append = FALSE, quote = TRUE, sep = " ",
            eol = "\n", na = "NA", dec = ".", row.names = TRUE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")
browseKEGG(KEGG,"hsa04022")#Select one of the hsa04022 pathways to display
