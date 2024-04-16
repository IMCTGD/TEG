import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from statistics import mean, stdev

import argparse

def list_std(list_data):
    mean_values = mean(list_data)
    std_devs = stdev(list_data)
    for i in range(len(list_data)):
        list_data[i] = (list_data[i] - mean_values) / std_devs
    return list_data


parser = argparse.ArgumentParser(description='read_atten_matix')
parser.add_argument('--chr_num', type=int, default='1')
parser.add_argument('--p_value', type=float, default='0.0001')
args = parser.parse_args()

# take a chromosome number
chr = args.chr_num
print("chr:", chr)



# Quadratic cross-validation, so there are four files
matrix_num = 4
cls_attn_list = []
for i in range(4):
    fined_ave_attn_weight = np.loadtxt('./matrix_snp/matrix{}-{}.csv'.format(chr,i), delimiter=',')
    # Extract the first row and normalize it
    cls_attn_list.append(list_std(fined_ave_attn_weight[0]))

cls_attn = [sum(items) for items in zip(*cls_attn_list)]







# Use the normal distribution to inscribe
# z_scores = stats.norm.ppf(q=1 - 0.0001, loc=np.mean(cls_attn), scale=np.std(cls_attn))
# z_scores = stats.norm.ppf(q=1 - 0.05, loc=np.mean(cls_attn), scale=np.std(cls_attn))

z_scores = stats.norm.ppf(q=1 - args.p_value, loc=np.mean(cls_attn), scale=np.std(cls_attn))

significant_positions = np.where(cls_attn > z_scores)[0]
significant_num = len(significant_positions)
print("The total number of significant loci is",significant_num)
print("p_value <", args.p_value)

fig = plt.figure(figsize=(12, 4))

# Mapping the Distribution of Attention
plt.bar(range(len(cls_attn)), cls_attn)

plt.xlabel('Site number')
plt.ylabel('Attention Score')
plt.title('Attention Score of [CLS] Token to Other Words')


sorted_attn = np.sort(cls_attn)[::-1]
# Obtained index of significant loci
top_loci_indices = np.argsort(cls_attn)[::-1][:significant_num]
print(top_loci_indices)


# index is converted to normal coding, starting from zero
top_loci_indices = top_loci_indices - 1
# Add an index to reflect the global position
hold_idx = (128 *  (chr-1)) + top_loci_indices
# Load Site Information File
data_list = []
with open('./data/snp_content_f_3_2816.txt') as f:
    for line in f:
        splits = line.split()
        if len(splits) > 2:
            # # Extract the third column
            column3 = splits[2]
            data_list.append(column3)
# Enter the index and print
for idx in hold_idx:
    print(data_list[idx])

plt.show()

