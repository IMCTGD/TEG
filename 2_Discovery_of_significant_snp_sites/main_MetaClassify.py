import os
import time

import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

from data_loader_tools import *



class MetaClassifier(nn.Module):
    def __init__(self, num_experts, expert_feature_size, auto_gate=True):
        super().__init__()

        self.meta_classifier = nn.Sequential(
            nn.LayerNorm(num_experts * expert_feature_size),
            nn.Linear(num_experts * expert_feature_size, 2)
        )
        # 加入门控层
        if auto_gate:
            self.gates = nn.Parameter(torch.ones(num_experts) / num_experts)
        else:
            self.gates = torch.ones(num_experts) / num_experts

    def forward(self, expert_outputs):
        # Weighting of experts
        weighted_inputs = []
        for i, expert_out in enumerate(expert_outputs):
            weighted_out = expert_out * self.gates[i]
            weighted_inputs.append(weighted_out)

        meta_input = torch.cat(weighted_inputs, dim=1)

        output = self.meta_classifier(meta_input)
        return output



seed=1
torch.manual_seed(seed)
np.random.seed(seed)
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print("now using device:",device)

def dataloader_pos(chr_pos, chr_num):
    pos_data = data_pos_transform_to_tensor(chr_pos, chr_num=chr_num)
    pos_data = get_pos_embedding(pos_data, 512, device=pos_data.device)
    return pos_data

def get_chr_input(input, chr_num):
    # Extract string data and tags
    index = (chr_num - 1) * 128
    chr_input = input[:, index:(index + 128)]
    return chr_input

data = np.loadtxt('./data/genetype_f_2_2816.txt', delimiter=' ', dtype=str)
chr_pos = np.loadtxt('./data/snp_content_f_3_2816.txt', delimiter=' ', dtype=str)





from model.gwas_transformer_base_model import get_pos_embedding

from model.gwas_transformer_base_model import clsDNA


datasetX = 0

# num_experts = [1,2]
num_experts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

print("Execution-based classifiers:", num_experts)
print("datasetX:",datasetX)

net_chr = []


for i in num_experts:
    net_i = clsDNA(
    vocab = 40,
    seq_len= 128,
    num_classes = 2,
    dim = int(512),
    depth = 8,
    heads = 12,
    mlp_dim = 512,
    dropout = 0.,
    emb_dropout = 0.,
    get_last_feature= True
)
    net_i.to(device)
    print("Load model(chr)：", i)
    checkpoint_i = torch.load('./checkpoint/ADNI_model/chr{}/'.format(i) +'clsDNA'+'dataset-{}-ckpt.t_high_auc_model'.format(
               datasetX))
    net_i.load_state_dict(checkpoint_i['model'])
    net_i.eval()
    net_chr.append(net_i)

pos_datas = []

x_data, y_data = data_xy_transform_to_tensor_all(data)
x_data = x_data.to(torch.float32)
train_loader, val_loader, test_loader = createSSSDataset(x_data, y_data, seed=1, batch_size=12, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, datasetX=datasetX)


for i in num_experts:
    pos_data = dataloader_pos(chr_pos=chr_pos, chr_num=i)
    pos_datas.append(pos_data)

meta_classifier = MetaClassifier(num_experts=len(num_experts), expert_feature_size=512, auto_gate=False)
meta_classifier = meta_classifier.to(device)


n_epochs = 50

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(meta_classifier.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, last_epoch=-1)
##### Training
# Mixing accuracy
scaler = torch.cuda.amp.GradScaler(enabled=False)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    meta_classifier.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (input_data_row, target_row) in enumerate(train_loader):
        expert_outs = []
        targets = []
        for i in range(len(num_experts)):
            input_data = get_chr_input(input=input_data_row, chr_num=num_experts[i])
            targets.append(target_row)
            _, cls = net_chr[i](input_data.to(device), pos_datas[i].to(device))
            expert_outs.append(cls)

        targets = targets[0].to(device)

        outputs = meta_classifier(expert_outs)
        loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print("Training set accuracy：", 100. * correct / total)
    print("Training Set Loss: ", train_loss / (batch_idx + 1))
    return train_loss / (batch_idx + 1), meta_classifier.gates

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

min_loss = -1
high_auc = -1
high_acc = -1

##### Validation
def Val():
    global best_acc
    global min_loss
    global high_auc
    global high_acc
    meta_classifier.eval()
    val_loss = 0
    correct = 0
    total = 0
    prob_all = []
    prob_all_AUC = []
    label_all = []
    with torch.no_grad():
        for batch_idx, (input_data_row, target_row) in enumerate(val_loader):
            expert_outs = []
            targets = []
            for i in range(len(num_experts)):
                input_data = get_chr_input(input=input_data_row, chr_num=num_experts[i])
                targets.append(target_row)
                _, cls = net_chr[i](input_data.to(device), pos_datas[i].to(device))
                expert_outs.append(cls)
            targets = targets[0].to(device)
            outputs = meta_classifier(expert_outs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Calculate F1
            prob = outputs.cpu()
            labels = targets.cpu()
            prob = prob.numpy()  # Convert prob to CPU first, then to numpy, you don't need to convert to CPU first if you train on CPU itself
            prob_all.extend(np.argmax(prob, axis=1))  # Find the maximum index of each row
            label_all.extend(labels)
            # Calculate AUC
            prob_all_AUC.extend(prob[:, 1])  # prob[:,1] returns the number in the second column of each row. According to the parameters of this function, y_score represents the score of the larger labeled class, and therefore the value corresponding to the maximum index, not the maximum index value.


    print("val_loss:",val_loss)
    # print F1-score
    f1 = f1_score(label_all, prob_all)
    AUC = roc_auc_score(label_all, prob_all_AUC)
    print("F1-Score:{:.4f}".format(f1))
    print("AUC:{:.4f}".format(AUC))

    acc = 100. * correct / total
    print("val_acc:",acc)

    if min_loss == -1:
        min_loss = val_loss
    if high_auc == -1:
        high_auc = AUC
    if high_acc == -1:
        high_acc = acc
    if acc > high_acc:
        high_acc = acc
    if val_loss < min_loss:
        min_loss = val_loss



    if AUC >= high_auc:
        print('Saving the high AUC..')
        state = {"model": meta_classifier.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/meta_classifier/' + "meta_classifier" + 'ckpt.-{}-best_auc_model'.format(datasetX))
        high_auc = AUC


    print("min_loss:",min_loss)
    print("high_auc:", high_auc)
    print("val_high_ACC:",high_acc)


    return val_loss, acc, f1, AUC


##### Test
def test():
    meta_classifier.eval()
    correct = 0
    total = 0
    prob_all = []
    prob_all_AUC = []
    label_all = []
    with torch.no_grad():
        for batch_idx, (input_data_row, target_row) in enumerate(test_loader):
            expert_outs = []
            targets = []
            for i in range(len(num_experts)):
                input_data = get_chr_input(input=input_data_row, chr_num=num_experts[i])
                targets.append(target_row)
                _, cls = net_chr[i](input_data.to(device), pos_datas[i].to(device))
                expert_outs.append(cls)
            targets = targets[0].to(device)
            outputs = meta_classifier(expert_outs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 计算F1
            prob = outputs.cpu()
            labels = targets.cpu()
            prob = prob.numpy()
            prob_all.extend(np.argmax(prob, axis=1))
            label_all.extend(labels)
            # 计算AUC
            prob_all_AUC.extend(prob[:, 1])

    f1 = f1_score(label_all, prob_all)
    AUC = roc_auc_score(label_all, prob_all_AUC)
    print("test-F1-Score:{:.4f}".format(f1))
    print("test-AUC:{:.4f}".format(AUC))
    acc = 100. * correct / total
    print("test-ACC:",acc)
    return  acc, f1, AUC

list_loss = []
list_acc = []
list_trainloss = []
list_f1 = []
list_AUC = []

if __name__ == '__main__':
    list_trainloss = []
    for epoch in range(0, n_epochs):
        start = time.time()
        trainloss, gate_score = train(epoch)
        print(gate_score)
        scheduler.step()
        list_trainloss.append(trainloss)
        val_loss, acc, f1, AUC = Val()

        list_loss.append(val_loss)
        list_acc.append(acc)
        list_f1.append(f1)
        list_AUC.append(AUC)

    print("-----------------------------------------")
    print("training finished!")
    print("-----------------------------------------")
    # 加载 best checkpoint 文件
    print("loading best auc model...")
    checkpoint = torch.load('./checkpoint/meta_classifier/' + "meta_classifier" + 'ckpt.-{}-best_auc_model'.format(datasetX))
    meta_classifier.load_state_dict(checkpoint['model'])
    test_acc, test_f1, test_auc = test()
    print(meta_classifier.gates)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(list_trainloss, label='Train Loss')
    plt.plot(list_loss, label='Val Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(list_acc, label='Val Acc')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(list_f1, label='F1_score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(list_AUC, label='AUC')
    plt.legend()

    plt.show()

