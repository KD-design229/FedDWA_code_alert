---
noteId: "fb2eea40cfed11f0b2635d80fb35527a"
tags: []

---

# GPR 探地雷达联邦学习适配修改总结

## 一、模型架构修改

### 1. MobileViT 模型修改

**文件**: `model/MLModel.py`

**修改内容**: 在 `MobileViT` 类中新增 `gpr_mode` 参数

```python
# 原始版本
class MobileViT(nn.Module):
    def __init__(self, model_name='mobilevit_s', num_classes=10, pretrained=True):
        # 直接使用 timm 的 mobilevit
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

# GPR 适配版本 (gpr_mode=True)
class MobileViT(nn.Module):
    def __init__(self, model_name='mobilevit_s', num_classes=10, pretrained=True, gpr_mode=False):
        # 新增 GPR 预处理层
        if gpr_mode:
            self.gpr_preprocess = nn.Sequential(
                nn.InstanceNorm2d(3, affine=True),           # 可学习的信号归一化
                nn.Conv2d(3, 16, kernel_size=(5, 1), ...),   # 时间域卷积（垂直方向）
                nn.Conv2d(16, 16, kernel_size=(1, 5), ...),  # 空间域卷积（水平方向）
                nn.Conv2d(16, 3, kernel_size=1, ...),        # 融合回3通道
            )
        self.model = timm.create_model(...)
    
    def forward(self, x):
        if self.gpr_mode:
            x = self.gpr_preprocess(x)  # GPR 预处理
        return self.model(x)
```

**GPR 预处理层设计原理**:

| 组件 | 作用 | GPR 适配原因 |
|------|------|-------------|
| `InstanceNorm2d(affine=True)` | 可学习的信号归一化 | 不同设备/天线/土质导致信号强度差异大 |
| `Conv2d(kernel=(5,1))` | 时间域卷积（垂直） | 捕获深度方向的反射波特征 |
| `Conv2d(kernel=(1,5))` | 空间域卷积（水平） | 捕获横向延续性（管线、裂缝走向） |
| `Conv2d(kernel=1)` | 通道融合 | 将16通道特征融合回3通道送入MobileViT |

---

### 2. FedCLIP 模型修改

**文件**: `model/MLModel.py`

**新增类**: `GPRAdapter`

```python
class GPRAdapter(nn.Module):
    """GPR 专用 Adapter 模块"""
    def __init__(self, dim, reduction=4):
        hidden_dim = dim // reduction  # 512 → 128
        self.down_proj = nn.Linear(dim, hidden_dim)      # 下投影
        self.act = nn.GELU()                              # 激活
        self.up_proj = nn.Linear(hidden_dim, dim)        # 上投影
        self.scale = nn.Parameter(torch.ones(1) * 0.1)   # 可学习缩放
        
    def forward(self, x):
        return x + self.scale * self.up_proj(self.act(self.down_proj(x)))  # 残差连接
```

**FedCLIP 修改内容**:

```python
# 原始版本
class FedCLIP(nn.Module):
    def __init__(self, ...):
        # Adapter: MaskedMLP
        self.fea_attn = nn.Sequential(
            MaskedMLP(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            MaskedMLP(dim, dim),
            nn.Softmax(dim=1)
        )
        # 分类: Text Similarity
        logits = image_features @ text_features.t()

# GPR 适配版本 (gpr_mode=True)
class FedCLIP(nn.Module):
    def __init__(self, ..., gpr_mode=False):
        if gpr_mode:
            # GPR Adapter: 残差结构，更稳定
            self.fea_attn = nn.Sequential(
                GPRAdapter(dim, reduction=4),
                nn.LayerNorm(dim),
                GPRAdapter(dim, reduction=4),
                nn.LayerNorm(dim),
            )
            # GPR 分类头: 线性分类，不依赖 text encoder
            self.gpr_classifier = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim // 2, num_classes),
            )
        # GPR 专用 prompt
        prompts = [f"a ground penetrating radar image showing {c}" for c in class_names]
```

**GPR 模式设计原理**:

| 组件 | 原始版本 | GPR 版本 | 原因 |
|------|---------|----------|------|
| Adapter | MaskedMLP + Softmax | GPRAdapter + LayerNorm | 残差结构更稳定，避免梯度消失 |
| 分类方式 | Text Similarity | 线性分类头 | GPR 图像与自然图像差异大，text encoder 可能不适用 |
| Prompt | "a picture of {c}" | "a ground penetrating radar image showing {c}" | GPR 专用描述 |

---

### 3. 新增独立模型 GPR-FedSense

**文件**: `model/MLModel.py`

**新增类**:

```python
# 1. GPRSignalNorm - 信号归一化层
class GPRSignalNorm(nn.Module):
    """可学习的信号归一化，适配不同设备"""
    gamma, beta  # 可学习缩放和偏移
    gain         # 可学习增益

# 2. GPRFeatureExtractor - 特征提取器
class GPRFeatureExtractor(nn.Module):
    """时空分离的双分支特征提取"""
    signal_norm   # 信号归一化
    time_conv     # (5,1) 垂直卷积
    spatial_conv  # (1,5) 水平卷积
    fusion        # 特征融合

# 3. GPRFedModel - 完整模型
class GPRFedModel(nn.Module):
    """三层解耦架构"""
    local_extractor   # 本地私有层（不聚合）
    shared_backbone   # 共享层（全局聚合）
    classifier        # 个性化分类头
```

---

## 二、数据增强修改

**文件**: `utils/dataset.py`

**函数**: `get_gpr_transforms()`

### 原始增强策略

```python
def get_gpr_transforms(is_train=True, image_size=32, use_clip_norm=False):
    if is_train:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(...),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(...),
            A.ShiftScaleRotate(rotate_limit=0, ...),  # 禁止旋转
            A.OneOf([
                A.GaussNoise(...),
                A.CoarseDropout(...),
                A.MultiplicativeNoise(...),
            ], p=0.4),
            A.RandomBrightnessContrast(...),
            A.Normalize(...),
            ToTensorV2(),
        ])
```

### GPR 高级增强策略 (enable_advanced_gpr=True)

```python
def get_gpr_transforms(is_train=True, image_size=32, use_clip_norm=False, enable_advanced_gpr=True):
    if is_train:
        transforms_list = [
            # 基础变换（同原始）
            A.LongestMaxSize(...),
            A.PadIfNeeded(...),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(...),
            A.ShiftScaleRotate(rotate_limit=0, ...),
        ]
        
        # 【新增】GPR 高级增强
        if enable_advanced_gpr:
            transforms_list.extend([
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # 模拟天线耦合变化
                A.CLAHE(clip_limit=2.0, p=0.3),              # 模拟介电常数变化
            ])
        
        transforms_list.extend([
            # 噪声增强（同原始）
            A.OneOf([...], p=0.4),
            A.RandomBrightnessContrast(...),
            A.Normalize(...),
            ToTensorV2(),
        ])
```

### 数据增强设计原理

| 增强方法 | 代码 | GPR 适配原因 |
|---------|------|-------------|
| **弹性变换** | `A.ElasticTransform(alpha=20, sigma=6)` | 模拟地下介质不均匀导致的波形扭曲 |
| **禁止旋转** | `A.ShiftScaleRotate(rotate_limit=0)` | GPR 垂直轴是深度/时间，旋转会破坏物理意义 |
| **高斯噪声** | `A.GaussNoise(var_limit=(5, 30))` | 模拟现场电磁干扰 |
| **乘性噪声** | `A.MultiplicativeNoise(multiplier=[0.9, 1.1])` | 模拟设备增益波动 |
| **小范围遮挡** | `A.CoarseDropout(max_holes=4, max_height=15)` | 模拟信号缺失/饱和区域 |
| **RandomGamma** | `A.RandomGamma(gamma_limit=(80, 120))` | 模拟天线耦合变化（信号强度衰减） |
| **CLAHE** | `A.CLAHE(clip_limit=2.0)` | 模拟不同土质介电常数变化（局部对比度增强） |

---

## 三、命令行参数修改

**文件**: `main.py`

### 新增参数

```python
# GPR 专用参数
parser.add_argument('--gpr_mode', action='store_true',
    help='Enable GPR-specific model adaptations')
parser.add_argument('--enable_advanced_gpr', action='store_true',
    help='Enable advanced GPR data augmentation')
parser.add_argument('--gpr_backbone', type=str, default='cnn',
    choices=['cnn', 'resnet18', 'mobilevit'],
    help='Backbone for GPR-FedSense model')

# 模型选项新增
parser.add_argument('--model', choices=[..., 'gpr_fed'])
```

---

## 四、使用方法对比

### 实验 1: 原始 MobileViT（基线）

```bash
!python main.py \
    --dataset gpr_custom \
    --model mobilevit_s \
    --alg feddwa \
    --client_num 5 \
    --Tg 50
```

### 实验 2: MobileViT + GPR 模式

```bash
!python main.py \
    --dataset gpr_custom \
    --model mobilevit_s \
    --gpr_mode \
    --enable_advanced_gpr \
    --alg feddwa \
    --client_num 5 \
    --Tg 50
```

### 实验 3: 原始 FedCLIP（基线）

```bash
!python main.py \
    --dataset gpr_custom \
    --model fedclip \
    --alg feddwa \
    --client_num 5 \
    --Tg 50
```

### 实验 4: FedCLIP + GPR 模式

```bash
!python main.py \
    --dataset gpr_custom \
    --model fedclip \
    --gpr_mode \
    --enable_advanced_gpr \
    --alg feddwa \
    --client_num 5 \
    --Tg 50
```

### 实验 5: GPR-FedSense（全新架构）

```bash
!python main.py \
    --dataset gpr_custom \
    --model gpr_fed \
    --gpr_backbone cnn \
    --enable_advanced_gpr \
    --alg feddwa \
    --client_num 5 \
    --Tg 50
```

---

## 五、文件修改汇总

| 文件 | 修改类型 | 内容 |
|------|---------|------|
| `model/MLModel.py` | 修改 | `MobileViT` 类新增 `gpr_mode` 参数和 `gpr_preprocess` 层 |
| `model/MLModel.py` | 新增 | `GPRAdapter` 类 |
| `model/MLModel.py` | 修改 | `FedCLIP` 类新增 `gpr_mode` 参数、`GPRAdapter` 和 `gpr_classifier` |
| `model/MLModel.py` | 新增 | `GPRSignalNorm`, `GPRFeatureExtractor`, `GPRFedModel` 类 |
| `utils/dataset.py` | 修改 | `get_gpr_transforms()` 新增 `enable_advanced_gpr` 参数 |
| `utils/dataset.py` | 新增 | `A.RandomGamma` 和 `A.CLAHE` 增强 |
| `main.py` | 新增 | `--gpr_mode`, `--enable_advanced_gpr`, `--gpr_backbone` 参数 |
| `main.py` | 修改 | 模型初始化逻辑支持 GPR 模式 |

---

## 六、实验对比建议

| 对比维度 | 实验组 |
|---------|--------|
| **GPR 预处理效果** | MobileViT vs MobileViT+gpr_mode |
| **GPR Adapter 效果** | FedCLIP vs FedCLIP+gpr_mode |
| **数据增强效果** | 无增强 vs enable_advanced_gpr |
| **架构对比** | MobileViT vs FedCLIP vs GPR-FedSense |
| **联邦学习组件** | 基础 vs +FedVLS vs +FedDecorr vs +ALA |
