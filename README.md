# GWATransFormEnsemble: GWAS-based transformer and integrated learning to analyze genetic data to identify significant SNP loci and biological pathways
![Python 3.8.18](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)


![workflowa](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/5a34e2e3-251b-4729-bf3e-0065c8f5a552)
![workflowb](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/781effbb-1ff3-4ae5-a2ec-8d08682cd8ec)


This analysis process is mainly divided into three steps (the quality control of the data must be completed before these three steps),
1.the first step is to transform the genetic data into features; 
2.the second step is to construct a model and train it finally to find the significant snp loci and important chromosome sets; 
3.the third step is to do the analysis of the biological pathways based on the results of the second step.

## Representation of All Scenarios：This figure illustrates all possible scenarios. The representation format is as follows: for instance, AT1 represents A>T mutation with a count of 1.
![image](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/7352a7d1-457a-4156-8021-e6d367c9ffe9)



## Model Architecture of the Base Classifier：
![base_model](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/432a83d9-0291-41b8-903a-1cb2904c635a)


<br>The input consists of the genotype and the corresponding chromosomal position of each locus for the samples. The genotype input dimension is 40, which is expanded to 512 through word embedding. It is then combined with the transformed positional information and concatenated with a CLS token used for classification. The self-attention mechanism comprises 12 heads with an 8-layer structure. Subsequently, a fully connected layer of 512x2 is employed to generate binary classification probabilities (healthy individuals or patients).<br/>

## Model Architecture of the Meta-Model:
![meta_model](https://github.com/xuanan-zhu/GWAS_transformer/assets/84304647/6616c534-bec2-4d71-824b-6398e16b0163)


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

- # Usage example
- Train base classifiers chromosome by chromosome and sample set by sample set.
  
`python transformer_base_model_main.py --chr_num 1 --datasetX 0` 

`python base_model_vision.py --chr_num 1 --datasetX 0` # Calculate the self-attention score matrix and generate a CSV file.

`python read_atten_matix.py --chr_num 1 --p_value 0.0001` # Aggregate the four self-attention matrices for each chromosome (four-fold cross-validation) to obtain significant SNP loci.

`python read_atten_matix.py --chr_num 1 --p_value 0.0001` # Aggregate the four self-attention matrices for each chromosome (four-fold cross-validation) to obtain significant SNP loci.

`python main_MetaClassify.py --datasetX 1 --num_experts [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]` # Select base classifiers to construct the meta-model.




Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```
