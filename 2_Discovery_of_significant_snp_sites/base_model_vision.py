import numpy as np
import torch
from visualizer import get_local
import matplotlib.pyplot as plt
from tqdm import tqdm
get_local.activate()


from model.gwas_transformer_base_model import clsDNA , get_pos_embedding
from data_loader_tools import data_xy_transform_to_tensor, data_pos_transform_to_tensor, createSSSDataset

def all_layers_heads_ave(attention_maps):
    average_head_att_map = []
    for i in range(len(attention_maps)):
        att_map = attention_maps[i].squeeze()
        average_head_att_map.append(att_map.mean(axis=0))
    average_all_att_map = np.mean(average_head_att_map, axis=0)
    return average_all_att_map

# 1. Load models of the same structure
model = clsDNA(
    vocab = 40,
    seq_len= 128,
    num_classes = 2,
    dim = int(512),
    depth = 8,
    heads = 12,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    get_last_feature= False
)
seed = 1
device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
print("now using device:",device)

# 2. Selection of chromosomes and data sets
chr_num = 22
datasetX = 3
print("datasetX:",datasetX)

# 3. Load the checkpoint file
checkpoint = torch.load('./checkpoint/ADNI_model/chr{}/'.format(chr_num) +'clsDNA'+'dataset-{}-ckpt.t_high_auc_model'.format(
               datasetX))

# 4. Loading the weights and parameters of the model
model.load_state_dict(checkpoint['model'])

# 5. Load experimental data
chr_pos = np.loadtxt('./data/snp_content_f_3_2816.txt', delimiter=' ', dtype=str)
pos_data = data_pos_transform_to_tensor(chr_pos, chr_num=chr_num)
pos_data = pos_data.to(device)
pos_data = get_pos_embedding(pos_data, 512, device= pos_data.device)
data = np.loadtxt('./data/genetype_f_2_2816.txt', delimiter=' ', dtype=str)

print("the chromosome being tested now is：", chr_num)
x_data, y_data = data_xy_transform_to_tensor(data, chr_num=chr_num)

# 6. Construct the dataset, keeping the method of dividing the dataset the same as during training
train_loader, val_loader,test_loader = createSSSDataset(x_data, y_data, seed=seed, batch_size=1, train_ratio = 0.6, val_ratio = 0.2, test_ratio=0.2, datasetX=datasetX)
vis_loader = val_loader

model = model.to(device)
model.eval()

with torch.no_grad():
    ave_attn_weights = None
    sample_num = 0
    correct = 0
    for _, (inputs, targets) in enumerate(tqdm(vis_loader, desc="Processing")):
        get_local.clear()
        inputs = inputs.to(device)
        outputs, _ = model(inputs, pos_data)
        _, predicted = outputs.max(1)
        sample_num += len(targets)
        correct += predicted.eq(int(targets)).sum().item()
        cache = get_local.cache
        attention_maps = cache['Attention.forward']
        if ave_attn_weights is None:
            ave_attn_weights = all_layers_heads_ave(attention_maps)
        else:
            ave_attn_weights += all_layers_heads_ave(attention_maps)
        # break

    print("accuracy:",correct/sample_num)
    fined_ave_attn_weight = ave_attn_weights/sample_num
    # np.savetxt('./matrix_snp_PD/matrix{}-{}.csv'.format(chr_num,datasetX), fined_ave_attn_weight, delimiter=',')
    plt.imshow(fined_ave_attn_weight, cmap='viridis')  # cmap parameter sets the color mapping, 'viridis' is a common color mapping
    plt.colorbar()  # Add a color bar to show how the values correspond to the colors
    plt.title('Heatmap')  # Add title
    plt.show()