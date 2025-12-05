import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torchvision.datasets import DatasetFolder, ImageFolder
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = imagefolder_obj.samples[self.dataidxs]
        else:
            self.samples = imagefolder_obj.samples

        for image in self.samples:
            if self.transform is not None:
                self.data.append(self.transform(self.loader(image[0])).numpy())
            else:
                self.data.append(self.loader(image[0]))
            self.targets.append(int(image[1]))
        self.indices = range(len(self.targets))

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)



"""Just for different privacy,especially for DPSGD method"""
class RandomSampledDataset2(Dataset):
    def __init__(self, dataset, q=1.0):
        self.dataset = dataset
        self.indexes = range(len(self.dataset))
        self.length = int(len(self.indexes) * q)
        self.count = 0
        self.random_indexes = np.random.choice(self.indexes, self.length, replace=False)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image, label = self.dataset[self.random_indexes[item]]
        self.count += 1
        if self.count == self.length:
            self.reset()
        return image, label

    def reset(self):
        self.count = 0
        self.random_indexes = np.random.choice(self.indexes, self.length, replace=False)



class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform


    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y


class CustomDataset(Dataset):
    def __init__(self, dataset, subset_transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label


class PartitionedDataset(Dataset):
    def __init__(self, dataset, indexes, subset_transform=None):
        self.dataset = dataset
        self.indexes = list(indexes)
        self.subset_transform = subset_transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        image, label = self.dataset[self.indexes[item]]

        if self.subset_transform:
            image = self.subset_transform(image)

        return image, label

def load_dataset(name: str, sample_rate=1.0, data_dir='./data', resize=32):

    if name == 'cifar100tpds':
        return load_cifar100_liketpds(sample_rate)
    elif name == 'cifar10tpds':
        return load_cifar_liketpds(sample_rate)
    elif name == 'cinic-10':
        return load_cinic_10()
    elif name == 'tiny_ImageNet':
        return load_tiny_imagenet()
    elif name == 'gpr_custom':
        return load_custom_dataset(data_dir, sample_rate, resize)
    else:
        raise NotImplementedError


def load_custom_dataset(data_dir, sample_rate=1.0, resize=32):
    # 使用 Albumentations 定义增强流程
    # image_size 现在由参数控制
    image_size = resize 
    
    # [Modified] 如果尺寸是 224 (通常用于 CLIP/ViT/ResNet 等预训练模型)，建议使用 ImageNet/CLIP 的归一化参数
    # 否则使用 GPR 数据集原本计算的统计值
    use_clip_norm = (image_size == 224)
    
    train_transform = get_gpr_transforms(is_train=True, image_size=image_size, use_clip_norm=use_clip_norm)
    test_transform = get_gpr_transforms(is_train=False, image_size=image_size, use_clip_norm=use_clip_norm)

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        dataset_train = GPR_ImageFolder(train_dir, transform=train_transform)
        dataset_test = GPR_ImageFolder(test_dir, transform=test_transform)
    else:
        # Fallback: 自动处理未划分的数据集
        
        # 1. 检测是否为 Client 结构 (例如 client_0, client_1...)
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        client_dirs = [d for d in subdirs if 'client' in d.lower()]
        
        if len(client_dirs) > 0:
            print(f"检测到客户端结构数据 ({len(client_dirs)} clients). 正在合并并进行全局 8:2 划分...")
            
            # 扫描所有客户端文件夹以获取所有类别名称
            classes = set()
            for c_dir in client_dirs:
                c_path = os.path.join(data_dir, c_dir)
                cls_in_client = [d for d in os.listdir(c_path) if os.path.isdir(os.path.join(c_path, d))]
                classes.update(cls_in_client)
            
            classes = sorted(list(classes))
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            print(f"识别到的类别: {classes}")
            
            # 收集所有图片样本
            samples = []
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            for c_dir in client_dirs:
                c_path = os.path.join(data_dir, c_dir)
                for target_class in classes:
                    class_dir = os.path.join(c_path, target_class)
                    if not os.path.isdir(class_dir): continue
                    
                    for root, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
                        for fname in sorted(fnames):
                            if fname.lower().endswith(valid_exts):
                                path = os.path.join(root, fname)
                                item = (path, class_to_idx[target_class])
                                samples.append(item)
            
            # 创建一个临时的完整数据集
            # 我们初始化一个空的 GPR_ImageFolder，然后手动覆盖它的样本列表
            # 这样可以避免 ImageFolder 错误的扫描逻辑
            full_dataset = GPR_ImageFolder(data_dir, transform=None)
            full_dataset.samples = samples
            full_dataset.targets = [s[1] for s in samples]
            full_dataset.classes = classes
            full_dataset.class_to_idx = class_to_idx
            
            print(f"共加载 {len(samples)} 张图片。")
            
        else:
            # 2. 假设是扁平结构 (data_dir/class_x/...)
            full_dataset = GPR_ImageFolder(data_dir, transform=None)

        # --- 统一的划分逻辑 ---
        total_len = len(full_dataset)
        train_size = int(0.8 * total_len)
        
        # 生成随机索引
        indices = torch.randperm(total_len).tolist()
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # 创建分别带有 训练增强 和 测试预处理 的完整数据集副本
        # 注意：这里我们必须手动把 samples 塞进去，因为重新初始化会再次扫描
        full_train_dataset = GPR_ImageFolder(data_dir, transform=train_transform)
        full_test_dataset = GPR_ImageFolder(data_dir, transform=test_transform)
        
        if len(client_dirs) > 0:
            # 如果是 client 结构，副本也需要覆盖 samples
            for ds in [full_train_dataset, full_test_dataset]:
                ds.samples = samples
                ds.targets = [s[1] for s in samples]
                ds.classes = classes
                ds.class_to_idx = class_to_idx
        
        # 使用 Subset 提取对应索引的数据
        dataset_train = torch.utils.data.Subset(full_train_dataset, train_indices)
        dataset_test = torch.utils.data.Subset(full_test_dataset, test_indices)

    return dataset_train, dataset_test


class GPR_ImageFolder(datasets.ImageFolder):
    """
    自定义 ImageFolder，使用 OpenCV 读取图像并支持 Albumentations 增强
    """
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Fail to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # Albumentations expects 'image' key
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

def get_gpr_transforms(is_train=True, image_size=32, use_clip_norm=False, enable_advanced_gpr=True):
    """
    参考用户提供的 Notebook 实现的数据增强策略 (aug_methon == 1)
    
    Args:
        is_train: 是否为训练模式
        image_size: 图像尺寸
        use_clip_norm: 是否使用 CLIP 归一化参数
        enable_advanced_gpr: 是否启用 GPR 高级增强（时频域模拟）
    """
    if use_clip_norm:
        # CLIP / ImageNet 标准归一化参数
        # OpenAI CLIP: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        # Torchvision: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # 这里使用 OpenAI CLIP 的参数，因为我们用的是 CLIP 模型
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        # print(f"Using CLIP normalization for image_size={image_size}")
    else:
        # GPR 数据集自定义统计值
        # Mean (RGB): [0.49715162577510535, 0.49943902106757904, 0.5020123172567167]
        # Std  (RGB): [0.15914664258290495, 0.15962691844790716, 0.15891394991650823]
        mean = [0.49715162577510535, 0.49943902106757904, 0.5020123172567167]
        std = [0.15914664258290495, 0.15962691844790716, 0.15891394991650823]

    if is_train:
        # GPR 数据增强策略列表
        transforms_list = [
            # --- 尺寸与几何变换 ---
            # 步骤 1: 使用 RandomResizedCrop 替代原来的固定缩放和中心裁剪。
            # (这里为了保持一致性，使用 LongestMaxSize + PadIfNeeded，对应 notebook 中的逻辑)
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=134, p=1.0),

            # 步骤 2: 水平翻转（GPR 扫描线方向可以翻转）
            A.HorizontalFlip(p=0.5),

            # 步骤 3: 弹性变换，模拟地下介质不均匀导致的波形轻微扭曲
            A.ElasticTransform(p=0.5, alpha=20, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.ShiftScaleRotate(
                shift_limit=0.05,      # 轻微平移 (±5%)
                scale_limit=0.1,       # 轻微缩放 (±10%)
                rotate_limit=0,        # ❌ GPR数据不应旋转! (深度-时间轴有物理意义)
                border_mode=0,         # 黑色填充
                p=0.3                  # 降低概率
            ),
        ]
        
        # 高级 GPR 增强（模拟现场环境变化）
        if enable_advanced_gpr:
            transforms_list.extend([
                # 模拟天线耦合变化（信号强度衰减）
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                
                # 模拟不同土质的介电常数变化（对比度）
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            ])
        
        # 继续添加通用增强
        transforms_list.extend([
            # --- 噪声与遮挡组合 ---
            # 步骤 4: 使用 OneOf 组合器模拟不同类型的现场干扰
            A.OneOf([
                # 选项1: 高斯噪声（电磁干扰）
                A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
                
                # 选项2: 小范围随机遮挡（信号缺失/饱和）
                A.CoarseDropout(
                    max_holes=4,
                    max_height=15,
                    max_width=15, 
                    min_holes=2,
                    fill_value=0, 
                    p=1.0
                ),
                
                # 选项3: 乘性噪声（增益波动）
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=1.0),
            ], p=0.4),

            # --- 信号强度与颜色变换 ---
            # 步骤 5: 随机调整亮度和对比度（模拟不同深度的信号衰减）
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15, 
                p=0.5
            ),
            
            # --- 标准化与格式转换 ---
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        
        return A.Compose(transforms_list)
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=134, p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])



def load_cinic_10(data_dir='./data/cinic-10/', sample_rate=1.0):
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    transform_cinic_10_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std),
    ])
    transform_cinic_10_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std),
    ])
    dl_obj = ImageFolder_custom
    dataset_train = dl_obj(data_dir + 'train/', transform=transform_cinic_10_train)
    dataset_test = dl_obj(data_dir + 'test/', transform=transform_cinic_10_test)
    return dataset_train, dataset_test




def load_tiny_imagenet(data_dir='./data/tiny-imagenet-200/', sample_rate=1.0):
    transform_tiny_imagenet_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_tiny_imagenet_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # dataset_train = datasets.ImageNet(data_dir, train=True, download=True, transform=transform_tiny_imagenet_train)
    # dataset_test = datasets.ImageNet(data_dir, train=False, download=True, transform=transform_tiny_imagenet_test)
    # print('dataset_train[0][1]=',dataset_train.__getitem__(0)[0].shape)
    dl_obj = ImageFolder_custom
    dataset_train = dl_obj(data_dir + 'train/', transform=transform_tiny_imagenet_train)
    dataset_test = dl_obj(data_dir + 'val/', transform=transform_tiny_imagenet_test)
    return dataset_train, dataset_test





def load_cifar100_liketpds(sample_rate=1.0):
    train_data = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    test_data = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    # train_idcs = np.random.RandomState(seed=42).permutation(len(train_data))
    # train_idcs = train_idcs[:int(sample_rate * len(train_data))]
    # test_idcs = np.random.RandomState(seed=42).permutation(len(test_data))
    # test_idcs = test_idcs[:int(sample_rate * len(test_data))]
    # train_set = CustomSubset(train_data, train_idcs)
    # test_set = CustomSubset(test_data, test_idcs)

    return train_data, test_data

def load_cifar_liketpds(sample_rate=1.0):
    train_data = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    test_data = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    # train_idcs = np.random.RandomState(seed=42).permutation(len(train_data))
    # train_idcs = train_idcs[:int(sample_rate * len(train_data))]
    # test_idcs = np.random.RandomState(seed=42).permutation(len(test_data))
    # test_idcs = test_idcs[:int(sample_rate * len(test_data))]
    # train_set = CustomSubset(train_data, train_idcs)
    # test_set = CustomSubset(test_data, test_idcs)

    return train_data, test_data
