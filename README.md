# GWATransFormEnsemble: GWAS-based transformer and integrated learning to analyze genetic data to identify significant SNP loci and biological pathways
![Python 3.8.18](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

![image](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/6b59218d-2f9b-4b54-84c2-cd440bb21582)
![image](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/7b4407f7-4e8e-4b36-ade5-a1a6db1e092f)
[workflowa.pdf](https://github.com/xuanan-zhu/GWAS_transformer/files/14692215/workflowa.pdf)
[workflowb.pdf](https://github.com/xuanan-zhu/GWAS_transformer/files/14692231/workflowb.pdf)


This analysis process is mainly divided into three steps (the quality control of the data must be completed before these three steps),
1.the first step is to transform the genetic data into features; 
2.the second step is to construct a model and train it finally to find the significant snp loci and important chromosome sets; 
3.the third step is to do the analysis of the biological pathways based on the results of the second step.

## Representation of All Scenarios：This figure illustrates all possible scenarios. The representation format is as follows: for instance, AT1 represents A>T mutation with a count of 1.
![image](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/0cbc2cb3-04ad-4dc9-a7bb-58e2059ea51e)


## Model Architecture of the Base Classifier：
![image](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/5bfa5053-250d-4f82-9fd6-07cb965b40aa)
[base_model.pdf](https://github.com/xuanan-zhu/GWAS_transformer/files/14692232/base_model.pdf)

<br>The input consists of the genotype and the corresponding chromosomal position of each locus for the samples. The genotype input dimension is 40, which is expanded to 512 through word embedding. It is then combined with the transformed positional information and concatenated with a CLS token used for classification. The self-attention mechanism comprises 12 heads with an 8-layer structure. Subsequently, a fully connected layer of 512x2 is employed to generate binary classification probabilities (healthy individuals or patients).<br/>

## Model Architecture of the Meta-Model:
![image](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/bea58714-8751-4057-8070-32400ac80e8c)
[meta_model.pdf](https://github.com/xuanan-zhu/GWAS_transformer/files/14692234/meta_model.pdf)

<br>CLS represents the output CLS token from the preceding layer of the fully connected layer in the base classifier. It is combined and inputted into a single layer of fully connected neurons to produce binary classification probabilities(healthy individuals or patients).<br/>


## Requirements

- Python 3.8.18
- einops==0.7.0
- matplotlib==3.7.2
- numpy==1.24.3
- scikit_learn==1.3.2
- scipy==1.12.0
- torch==2.1.2+cu118
- tqdm==4.66.1
- visualizer==0.0.10

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```
