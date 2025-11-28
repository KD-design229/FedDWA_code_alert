import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# [ 🛡️ 基础设施: Subset 兼容性补丁 ]
def get_targets_safe(dataset):
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    if isinstance(dataset, Subset):
        if hasattr(dataset.dataset, 'targets'):
            return np.array(dataset.dataset.targets)[dataset.indices]
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    targets = []
    for _, y in loader:
        targets.extend(y.numpy())
    return np.array(targets)

def noniid_type8(datasetname, dataset, num_users, num_classes=10, sample_assignment=None, test=False, logger=None):
    dataset_image = []
    dataset_label = []
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)

    for _, data in enumerate(dataloader, 0):
        dataset_data, dataset_targets = data
    
    dataset_image.extend(np.array(dataset_data))
    dataset_label.extend(np.array(dataset_targets))
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_shards = int(num_users * 2)
    order = np.argsort(dataset_label)
    x_sorted = dataset_image[order]
    y_sorted = dataset_label[order]

    n_shards = num_users * 2
    x_shards = np.array_split(x_sorted, n_shards)
    y_shards = np.array_split(y_sorted, n_shards)
    
    if sample_assignment is None:
        sample_assignment = np.array_split(np.random.permutation(n_shards), num_users)

    data = []
    for w in range(num_users):
        indices = sample_assignment[w]
        X = np.concatenate([x_shards[i] for i in indices])
        y = np.concatenate([y_shards[i] for i in indices])
        if logger: logger.info(np.unique(y))
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        data.append([(x, y) for x, y in zip(X, y)])
    return data, sample_assignment

def noniid_type9(datasetname, trainset, testset, num_users, num_classes=10, dirichlet_alpha=0.1, least_samples=20, logger=None):
    train_labels = get_targets_safe(trainset)
    test_labels = get_targets_safe(testset)
    
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)

    dataset_image = []
    for _, (data, _) in enumerate(trainloader): dataset_image.extend(np.array(data))
    for _, (data, _) in enumerate(testloader): dataset_image.extend(np.array(data))
    dataset_image = np.array(dataset_image)
    dataset_label = np.concatenate((train_labels, test_labels))

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    min_size = 0
    K = num_classes
    N = len(dataset_label)

    while min_size < least_samples:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(num_users):
        dict_users[j] = idx_batch[j]

    train_data, test_data = [], []
    for i in range(num_users):
        indices = list(dict_users[i])
        if logger: logger.info(f'Client {i} labels: {np.unique(dataset_label[indices])}')
        X_train, X_test, y_train, y_test = train_test_split(dataset_image[indices], dataset_label[indices], train_size=0.8, shuffle=True)
        train_data.append([(torch.tensor(x), torch.tensor(y)) for x, y in zip(X_train, y_train)])
        test_data.append([(torch.tensor(x), torch.tensor(y)) for x, y in zip(X_test, y_test)])
    return train_data, test_data

def noniid_type10(datasetname, dataset, num_users, num_types, ratio, num_classes=10, logger=None):
    """
    [ 🎓 博士课堂: 高保真重构版 ]
    采用灵活的分配策略，允许 num_types > num_users，
    确保所有数据都被使用，且严格遵守 num_types 的切分定义。
    """
    # 1. 快速加载数据
    trainloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)
    for _, train_data in enumerate(trainloader, 0):
        dataset_image, dataset_label = train_data

    # 2. 准备 "Dominant" (主要) 和 "Small" (剩余随机) 数据
    order = torch.randperm(dataset_image.shape[0])
    image_random = dataset_image[order]
    label_random = dataset_label[order]
    
    offset = int(dataset_image.shape[0] * ratio)
    image_class = image_random[:offset] # 主要部分 (80%)
    label_class = label_random[:offset]
    image_s = image_random[offset:]     # 剩余部分 (20%)
    label_s = label_random[offset:]

    # 3. 对主要数据按类别排序并切分为 num_types 份
    order = torch.argsort(label_class)
    x_sorted = image_class[order]
    y_sorted = label_class[order]
    
    # 强制切分为用户要求的 num_types 份 (比如 4 份)
    x_shards = torch.tensor_split(x_sorted, num_types)
    y_shards = torch.tensor_split(y_sorted, num_types)

    # 4. 初始化每个用户的容器
    # 使用列表而不是直接 tensor 连接，避免频繁内存拷贝
    x_client_buckets = [[] for _ in range(num_users)]
    y_client_buckets = [[] for _ in range(num_users)]

    # 5. [核心优化] 智能分配逻辑
    # 无论 shard 多还是人多，都公平分配
    for i in range(num_types):
        # 取出第 i 个 shard (代表某种数据分布)
        shard_x = x_shards[i]
        shard_y = y_shards[i]
        
        # 策略：如果人比 shard 多，这个 shard 要拆给多个人
        # 如果 shard 比人多，一个人要拿多个 shard (轮询)
        
        if num_users > num_types:
            # 这种情况下，一个 shard 要分给 (num_users / num_types) 个人
            # 这里的计算比较复杂，为了保持代码极其简洁且稳健，
            # 我们直接使用最通用的 "发牌" 模式：
            # 将 shard 再细分，填补空缺的用户
            
            # 计算当前 shard 应该覆盖哪些用户索引
            # 这是一个简化的映射，确保覆盖所有用户
            sub_chunks = int(np.ceil(num_users / num_types))
            shard_x_parts = torch.tensor_split(shard_x, sub_chunks)
            shard_y_parts = torch.tensor_split(shard_y, sub_chunks)
            
            for j in range(len(shard_x_parts)):
                target_user = (i * sub_chunks + j) % num_users
                x_client_buckets[target_user].append(shard_x_parts[j])
                y_client_buckets[target_user].append(shard_y_parts[j])
        else:
            # [用户遇到的情况: N=3, T=4]
            # 直接轮询分配：Shard 0->U0, Shard 1->U1, Shard 2->U2, Shard 3->U0
            target_user = i % num_users
            x_client_buckets[target_user].append(shard_x)
            y_client_buckets[target_user].append(shard_y)

    # 6. 分配剩余的 20% 随机数据 (均匀分配)
    x_split_all = torch.tensor_split(image_s, num_users)
    y_split_all = torch.tensor_split(label_s, num_users)

    # 7. 合并最终数据
    data = []
    for i in range(num_users):
        # 合并 Dominant 部分
        if len(x_client_buckets[i]) > 0:
            x_dom = torch.cat(x_client_buckets[i])
            y_dom = torch.cat(y_client_buckets[i])
        else:
            x_dom = torch.tensor([])
            y_dom = torch.tensor([])
            
        # 合并 Random 部分
        X = torch.cat((x_dom, x_split_all[i]))
        y = torch.cat((y_dom, y_split_all[i]))
        
        if logger: 
            logger.info(f'Client {i} label types: {torch.unique(y)}')

        data.append([(x, y) for x, y in zip(X, y)])

    return data

def dirichlet_noniid(dataset, num_users=10, dirichlet_alpha=100, sample_matrix_test=None, test=False):
    train_labels = get_targets_safe(dataset)
    class_num = train_labels.max() + 1
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(train_labels))
    idxs_labels = np.vstack((idxs, train_labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    class_lableidx = [idxs_labels[:, idxs_labels[1, :] == i][0, :] for i in range(class_num)]

    if test is True and sample_matrix_test is not None:
        sample_matrix = sample_matrix_test
    else:
        sample_matrix = np.random.dirichlet([dirichlet_alpha for _ in range(num_users)], class_num).T
    class_sampe_start = [0 for i in range(class_num)]
    for i in range(num_users):
        rand_set, class_sampe_start = sample_rand(sample_matrix[i], class_lableidx, class_sampe_start)
        dict_users[i] = rand_set
    return dict_users, sample_matrix

def sample_rand(rand, class_lableidx, class_sampe_start):
    class_sampe_end = [start + int(len(class_lableidx[sidx]) * rand[sidx]) for sidx, start in enumerate(class_sampe_start)]
    rand_set = np.array([], dtype=np.int32)
    for eidx, rand_end in enumerate(class_sampe_end):
        rand_start = class_sampe_start[eidx]
        if rand_end <= len(class_lableidx[eidx]):
            rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:rand_end]], axis=0)
        else:
            if rand_start < len(class_lableidx[eidx]):
                rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:]], axis=0)
            else:
                if len(class_lableidx[eidx]) > 0:
                     rand_set = np.concatenate([rand_set, random.sample(list(class_lableidx[eidx]), min(len(class_lableidx[eidx]), rand_end - rand_start + 1))], axis=0)
    if rand_set.shape[0] == 0:
        rand_set = np.concatenate([rand_set, class_lableidx[0][0:1]], axis=0)
    return rand_set, class_sampe_end

def data_loader(datasetname, trainset, testset):
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)
    dataset_image, dataset_label = [], []
    for _, (data, target) in enumerate(trainloader):
        dataset_image.extend(np.array(data))
        dataset_label.extend(np.array(target))
    for _, (data, target) in enumerate(testloader):
        dataset_image.extend(np.array(data))
        dataset_label.extend(np.array(target))
    return np.array(dataset_image), np.array(dataset_label)