# 导入必备工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 设置显示风格
plt.style.use('fivethirtyeight')

# 分别读取训练tsv和验证tsv
train_data = pd.read_csv("./cn_data/train.tsv", sep="\t")
valid_data = pd.read_csv("./cn_data/dev.tsv", sep="\t")

# 获得训练数据标签数量分布
sns.countplot(data=train_data)
plt.title("train_data")
plt.show()

# 获取验证数据标签数量分布
sns.countplot(data=valid_data)
plt.title("valid_data")
plt.show()
