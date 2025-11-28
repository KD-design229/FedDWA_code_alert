# Git 仓库操作指南

这份指南总结了如何将本地代码上传到 GitHub，以及如何在日常开发中更新和同步代码。

## 1. 首次关联远程仓库 (初始化)

如果您有一个本地项目，想要上传到一个新的 GitHub 仓库，请执行以下步骤：

```powershell
# 1. 初始化 Git (如果尚未初始化)
git init

# 2. 关联远程仓库地址
# 将 <URL> 替换为您的 GitHub 仓库地址
git remote add origin https://github.com/KD-design229/FedDWA_code_alert.git

# 如果提示 "error: remote origin already exists"，请使用 set-url 修改：
# git remote set-url origin https://github.com/KD-design229/FedDWA_code_alert.git

# 3. 重命名分支为 main (GitHub 默认主分支名)
git branch -M main

# 4. 首次推送代码
# -u 参数会将本地 main 分支与远程 main 分支关联，以后只需 git push
git push -u origin main
```

---

## 2. 日常更新代码 (最常用)

当您修改了代码（如 `main.py`）并想要保存到 GitHub 时，只需执行这三步：

```powershell
# 第一步：添加更改
# "." 代表添加当前目录下的所有更改
git add .

# 第二步：提交更改
# -m 后面是本次更新的说明，建议写清楚改了什么
git commit -m "修改了学习率参数"

# 第三步：推送到远程
# 因为首次已经关联过，这里不需要再输地址
git push
```

---

## 3. 在新电脑/新环境上工作

如果您换了一台电脑，或者在另一个文件夹中想要下载代码继续工作：

```powershell
# 1. 克隆 (下载) 仓库
# 这会自动下载代码并配置好远程关联，无需手动 git remote add
git clone https://github.com/KD-design229/FedDWA_code_alert.git

# 2. 进入项目文件夹
cd FedDWA_code_alert

# 3. 开始工作...
# 修改代码后，直接使用上面的 "日常更新代码" 三部曲即可：
# git add . -> git commit -> git push
```

## 4. 常用命令速查

| 命令 | 作用 |
| :--- | :--- |
| `git status` | 查看当前哪些文件被修改了，哪些还没提交 |
| `git log` | 查看提交历史记录 |
| `git pull` | 如果远程仓库有更新（比如别人推了代码），拉取到本地 |
| `git remote -v` | 查看当前关联的远程仓库地址 |
