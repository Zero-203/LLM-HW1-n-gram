# 中文 n-gram 语言模型实验

> GitHub 仓库链接占位符：<https://github.com/OWNER/REPOSITORY>。创建仓库后请替换为实际地址。

本项目使用中文分词语料训练 unigram 至 6-gram 语言模型，比较 add-k、Good-Turing 和 Katz 平滑，并根据验证集交叉熵选择配置，在测试集上报告结果。项目同时提供语料清洗、文章级数据划分、文本续写和结果绘图功能。

## 项目结构

```text
.
├── data/                 # 原始语料、清洗数据、数据集及划分清单
├── model/                # 六种阶数的词表、计数和元数据
├── result/
│   ├── continuation/     # 续写结果
│   ├── figures/          # 实验图表
│   ├── metrics/          # 分阶指标与模型对比 CSV
│   └── reports/          # 实验配置及报告
├── src/                  # 清洗、划分、训练评估、续写及绘图代码
├── tests/                # unittest 测试
├── n_gram.py             # 交互式入口
├── n-gram.pyproj         # Visual Studio Python 项目
├── n-gram.slnx           # Visual Studio 解决方案
└── requirements.txt      # Python 依赖
```

## 环境与使用

需要 Python 3.10 或更高版本。在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python n_gram.py
```

交互菜单提供以下流程：

1. 清洗 `data/raw/` 中的语料，生成 `data/interim/` 文件。
2. 按文章划分训练、验证和测试集，生成 `data/processed/` 文件。
3. 分别输入 `n=1` 至 `n=6` 训练、选择平滑参数并评估模型。
4. 选择阶数与文本文件进行续写；请先确保对应模型已生成。

生成实验图表：

```powershell
python src/auxiliary/plot_experiment_results.py
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

实验数据口径、语料来源说明及结果详见 [`result/reports/experiment_config.md`](result/reports/experiment_config.md) 和 [`result/reports/experiment_report.md`](result/reports/experiment_report.md)。