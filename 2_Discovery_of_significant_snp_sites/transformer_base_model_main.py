



import argparse
import os
import time

from torch import optim, nn
from data_loader_tools import *
from model.clsDNA import clsDNA
import numpy as np




parser = argparse.ArgumentParser(description='PyTorch snp-seek-base-transformer Training')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
parser.add_argument('--opt', default="adam") # sgd adam adamW
parser.add_argument('--net', default='clsDNA')
parser.add_argument('--bs', default='512')
parser.add_argument('--n_epochs', type=int, default='300')

args = parser.parse_args()

seed=1


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("now using device:",device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
min_loss = -1
high_auc = -1


# Data
print('==> Preparing data..')

data = np.loadtxt('./data/genetype_f_2_2816.txt', delimiter=' ', dtype=str)

chr_num = 22
datasetX = 0

print("Now training the chromosomes：", chr_num)
print("Now datasetx:",datasetX)
x_data, y_data= data_xy_transform_to_tensor(data, chr_num=chr_num)
x_data = x_data.to(torch.float32)

from model.clsDNA import get_pos_embedding


chr_pos = np.loadtxt('./data/snp_content_f_3_2816.txt', delimiter=' ', dtype=str)
pos_data = data_pos_transform_to_tensor(chr_pos, chr_num=chr_num)
pos_data = pos_data.to(device)
pos_data = get_pos_embedding(pos_data, 512, device= pos_data.device)

train_loader, val_loader, _ = createSSSDataset(x_data, y_data, seed=1, batch_size=42, train_ratio = 0.6, val_ratio = 0.2,
                                               test_ratio=0.2,datasetX=datasetX)

print("========================>finished loading")


if args.net=="clsDNA":
    # transformer for DNA
    net = clsDNA(
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


criterion = nn.CrossEntropyLoss()




if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
elif args.opt == "adamW":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)

# use cosine scheduling
use_lr_low = True
if use_lr_low:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, last_epoch=-1)

##### Training
# 混合精度
scaler = torch.cuda.amp.GradScaler(enabled=False)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=False):
            outputs, _ = net(inputs, pos_data)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("训练集准确率：", 100. * correct / total)
    print("训练集Loss: ", train_loss / (batch_idx + 1))
    return train_loss / (batch_idx + 1)

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


##### Validation
def Val():
    global best_acc
    global min_loss
    global high_auc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    prob_all = []
    prob_all_AUC = []
    label_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs, pos_data)

            # 计算F1
            prob = outputs.cpu()
            labels = targets.cpu()
            prob = prob.numpy()  # Convert prob to CPU first, then to numpy, you don't need to convert to CPU first if you train on CPU itself
            prob_all.extend(np.argmax(prob, axis=1))  # Find the maximum index of each row
            label_all.extend(labels)
            # 计算AUC
            prob_all_AUC.extend(prob[:, 1])  # prob[:,1] return to the second column of each row of the number, according to the parameters of the function can be seen, y_score represents the score of the larger label class, and therefore is the maximum index corresponding to the value of that value, rather than the maximum index value
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    #打印F1-score
    f1 = f1_score(label_all, prob_all)
    AUC = roc_auc_score(label_all, prob_all_AUC)
    print("F1-Score:{:.4f}".format(f1))
    print("AUC:{:.4f}".format(AUC))
    
    if min_loss == -1:
        min_loss = val_loss
    if high_auc == -1:
        high_auc = AUC
    print("val_loss:")
    print(val_loss)
    print("min_loss:")
    print(min_loss)
    print("high_auc:", high_auc)

    if AUC >= high_auc:
        print('Saving the high AUC..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ADNI_model/chr{}/'.format(chr_num) + args.net + 'dataset-{}-ckpt.t_high_auc_model'.format(datasetX))
        high_auc = AUC

    # Save checkpoint.
    acc = 100. * correct / total

    return val_loss, acc, f1, AUC


list_loss = []
list_acc = []
list_trainloss = []
list_f1 = []
list_AUC = []


net = net.to(device)
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc, f1, AUC = Val()

    if use_lr_low:
        scheduler.step()  # step cosine scheduling

    list_loss.append(val_loss)
    list_acc.append(acc)
    list_trainloss.append(trainloss)
    list_f1.append(f1)
    list_AUC.append(AUC)



import matplotlib.pyplot as plt

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

# plt.savefig('loss_acc.png')
plt.show()