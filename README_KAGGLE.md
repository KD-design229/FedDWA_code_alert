# Kaggle 运行指南

## 1. 上传代码
将整个项目文件夹上传到 Kaggle Dataset，或者直接在 Kaggle Notebook 中 `git clone` 你的仓库（如果已推送到 GitHub）。
推荐方式：将代码打包为 ZIP 上传为一个 Kaggle Dataset，然后挂载到 Notebook。或者直接在 Notebook 中 Upload Files。

## 2. 上传数据集
将你的探底雷达数据集（GPR）上传为 Kaggle Dataset。
假设数据集结构如下：
```
/kaggle/input/my-gpr-dataset/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   ...
    │   └── class_8/
    └── test/
        ├── class_1/
        ...
        └── class_8/
```
如果你的数据集没有分 train/test 文件夹，代码会自动按 8:2 比例划分。

## 3. 运行命令
在 Kaggle Notebook 中运行以下命令：

```bash
# 安装依赖
!pip install -r requirements.txt

# 运行训练
# 请将 /kaggle/input/my-gpr-dataset 替换为你实际的数据集路径
!python main.py \
    --dataset gpr_custom \
    --data_dir /kaggle/input/my-gpr-dataset \
    --model cnn \
    --client_num 4 \
    --gpu 0 \
    --num_classes 8 \
    --Tg 50 \
    --E 1 \
    --lr 0.01
```

## 4. 代码修改说明 (已自动完成)
为了适配你的数据集，我对代码做了以下修改：
1. **main.py**: 
   - 添加了 `--data_dir` 参数用于指定数据集路径。
   - 添加了 `gpr_custom` 数据集选项。
   - 设置 `gpr_custom` 对应的类别数为 8。
2. **servers/serverBase.py**: 
   - 将 `data_dir` 参数传递给数据加载函数。
3. **utils/dataset.py**: 
   - 新增 `load_custom_dataset` 函数，支持从指定文件夹加载图片。
   - 自动将图片缩放为 32x32 以适配 CNN 模型。
   - 支持自动划分训练集/测试集（如果未分文件夹）。
