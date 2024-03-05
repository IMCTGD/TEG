## extract snp site

# Load GWAS analysis results
results_log <- read.table("./logistic_results.assoc_2.logistic", head=TRUE)

sorted_result_log_index <- order(results_log$P, decreasing = FALSE)

sorted_results_log <- results_log[sorted_result_log_index, ]




# Going for 128 snp based on each chromosome 

library(dplyr)

# Sorted by CHR and P-value
sorted_results_log <- sorted_results_log %>%
  arrange(CHR, P)

# Group by CHR and select the first 128 rows of each CHR group
top_128_rows <- sorted_results_log %>%
  group_by(CHR) %>%
  slice_head(n = 128)

# Extract SNP columns
result <- top_128_rows$SNP
writeLines(result, "selected_snps.txt")
# Output results
print(result)

## deal vcf file
library(tidyverse)
library(vcfR)
library(do)
library(R.utils)

# Reading vcf files and fam files
vcf <- read.vcfR("./selected_snps_128_per_chr_vcf.vcf")
pheno <- read.table("./pheno12.fam")

fix <- vcf@fix
gt <- vcf@gt
meta <- vcf@meta


fix <- as.data.frame(fix)
gt <- as.data.frame(gt)
meta <- as.data.frame(meta)

gt <- gt[,2:length(colnames(gt))]
sample_n <- length(colnames(gt))
sample_name <- list(colnames(gt))[[1]]

# Extract the information from the previously selected snp loci
snp_index <- fix$ID %in% result
fix <- fix[snp_index,]
gt <- gt[snp_index,]


num2genetype <- function(gt, fix) {
  sample_n <- length(colnames(gt))
  sample_vec <- length(gt[,1])
  for (i in 1:sample_n) {
    print(i)
    for (j in 1:sample_vec) {
      if(is.na(gt[, i][j])){
        gt[,i][j]<- "N"
        next
      }
      if(fix$REF[j] == fix$ALT[j]){
        gt[,i][j]<- paste0(fix$REF[j],fix$ALT[j],"0")
        next
      }
      if(gt[,i][j] == "0/1"){
        gt[,i][j]<- paste0(fix$REF[j],fix$ALT[j],"1")
        next
      }
      if(gt[,i][j] == "1/0"){
        gt[,i][j]<- paste0(fix$REF[j],fix$ALT[j],"1")
        next
      }
      if(gt[,i][j] == "0/0"){
        gt[,i][j]<- paste0(fix$REF[j],fix$ALT[j],"0")
        next
      }
      if(gt[,i][j] == "1/1"){
        gt[,i][j]<- paste0(fix$REF[j],fix$ALT[j],"2")
        next
      }
      gt[,i][j]<- "N"
    }
  }
  return(gt)
}

gt <- num2genetype(gt = gt,fix = fix)

gt <- as.data.frame(t(gt))
colnames(gt) <- fix$ID
gt$pheno <- pheno$V6

check_list <- unique(unlist(gt))
print(check_list)


write.table(gt,file="./genetype_f_2_2816.txt",quote=F,col.name=F,row.names=F)
write.csv(gt,file="./detail_genetype_f_2_2816.csv",quote=F,row.names = T)

write.table(fix,file="./snp_content_f_3_2816.txt",quote=F,col.name=F,row.names=F)
write.csv(fix,file="./detail_snp_content_f_3_2816.csv",quote=F,row.names = T)



