
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
# 热编码函数

from collections import defaultdict
import torch


def one_hot_encode2(snp):
    encoding_table = {
        'AA0': 0, 'AT0': 1, 'AC0': 2, 'AG0': 3,
        'TA0': 4, 'TT0': 5, 'TC0': 6, 'TG0': 7,
        'CA0': 8, 'CT0': 9, 'CC0': 10, 'CG0': 11,
        'GA0': 12, 'GT0': 13, 'GC0': 14, 'GG0': 15,
        'AT1': 16, 'AC1': 17, 'AG1': 18,
        'TA1': 19, 'TC1': 20, 'TG1': 21,
        'CA1': 22, 'CT1': 23, 'CG1': 24,
        'GA1': 25, 'GT1': 26, 'GC1': 27,
        'AT2': 28, 'AC2': 29, 'AG2': 30,
        'TA2': 31, 'TC2': 32, 'TG2': 33,
        'CA2': 34, 'CT2': 35, 'CG2': 36,
        'GA2': 37, 'GT2': 38, 'GC2': 39
    }

    num_classes = len(encoding_table)
    encoded_snp = np.zeros(num_classes)
    if snp in encoding_table:
        encoded_snp[encoding_table[snp]] = 1
    return encoded_snp





def rename_numbers(input_list):
    unique_numbers = list(set(input_list))  # 获取列表中唯一的数字
    unique_numbers.sort()  # 对唯一数字进行排序

    number_to_index = {num: index for index, num in enumerate(unique_numbers)}  # 创建数字到索引的映射

    renamed_list = [number_to_index[num] for num in input_list]  # 使用映射将数字重命名
    return renamed_list


def data_xy_transform_to_tensor(data, chr_num=19):
    # 提取字符串数据和标签
    snp_data = data[:, :-1]
    index = (chr_num - 1) * 128
    snp_data = snp_data[:, index:(index + 128)]
    snp_data = snp_data.tolist()
    labels = data[:, -1].astype(int)
    labels = labels.tolist()
    labels = rename_numbers(labels)
    # 创建一个新的 NumPy 数组来存储编码后的数据
    encoded_snp_data = []
    # 对整个 SNP 数据数组进行编码，并将结果存储在 encoded_snp_data 中
    for i in range(len(snp_data)):
        encoded_snp_data.append([one_hot_encode2(snp) for snp in snp_data[i]])
    encoded_snp_data = np.array(encoded_snp_data)
    x_data = torch.from_numpy(encoded_snp_data)
    y_data = torch.tensor(labels)
    return x_data, y_data


def data_xy_transform_to_tensor_all(data):
    # 提取字符串数据和标签
    snp_data = data[:, :-1]
    snp_data = snp_data.tolist()
    labels = data[:, -1].astype(int)
    labels = labels.tolist()
    labels = rename_numbers(labels)
    # 创建一个新的 NumPy 数组来存储编码后的数据
    encoded_snp_data = []
    # 对整个 SNP 数据数组进行编码，并将结果存储在 encoded_snp_data 中
    for i in range(len(snp_data)):
        encoded_snp_data.append([one_hot_encode2(snp) for snp in snp_data[i]])
    encoded_snp_data = np.array(encoded_snp_data)
    x_data = torch.from_numpy(encoded_snp_data)
    y_data = torch.tensor(labels)
    return x_data, y_data


def data_pos_transform_to_tensor(pos, chr_num=19):
    print("pos:", chr_num)
    pos_data = pos[:, 1].astype(int)

    index = (chr_num - 1) * 128
    pos_data = pos_data[index:(index + 128)]

    pos_data = torch.from_numpy(pos_data)
    pos_data = pos_data.unsqueeze(1)

    return pos_data



def createSSSDataset(x_data, y_data, batch_size=42, seed=1, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                     datasetX=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(x_data, y_data)
    labels = y_data.numpy()
    # 统计每个类别的样本数量
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1

    # 计算每个类别在训练集和验证集中的样本数
    # train_ratio = 0.6
    # val_ratio = 0.2
    # test_ratio = 0.2
    train_samples_per_class = {label: int(count * train_ratio) for label, count in label_counts.items()}
    val_samples_per_class = {label: int(count * val_ratio) for label, count in label_counts.items()}
    test_samples_per_class = {label: int(count * test_ratio) for label, count in label_counts.items()}

    # 在每个类别中采样索引
    train_indices = []
    val_indices = []
    test_indices = []
    for label in label_counts:
        label_indices = np.where(labels == label)[0]
        i = datasetX
        n = train_samples_per_class[label] + val_samples_per_class[label] + test_samples_per_class[label]
        label_indices = np.random.choice(label_indices, n, replace=False)
        val_indice = label_indices[int(i * val_ratio * n):int((i + 1) * val_ratio * n)]
        train_indice1 = label_indices[:int(i * val_ratio * n)]
        train_indice2 = label_indices[int((i + 1) * val_ratio * n):n - int(test_ratio * n)]

        test_indice = label_indices[int(-test_ratio * n):]

        train_indices.extend(train_indice1)
        train_indices.extend(train_indice2)
        val_indices.extend(val_indice)
        test_indices.extend(test_indice)

    # 创建子数据集和数据加载器
    print("train_indices:",train_indices)
    print("val_indices:", val_indices)
    print("test_indices:", test_indices)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader








