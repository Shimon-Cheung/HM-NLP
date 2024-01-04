# 词汇表示成向量的无监督训练方法, 该过程将构建神经网络模型, 将网络参数作为词汇的向量表示, 它包含CBOW和skipgram两种训练模式
# CBOW(Continuous bag of words)模式
# 代码运行在python解释器中
# 导入fasttext
import fasttext

# 使用fasttext的train_unsupervised(无监督训练方法)进行词向量的训练
# 它的参数是数据集的持久化文件路径'data/fil9'
# model = fasttext.train_unsupervised('./fil9', "skipgram", dim=300, epoch=1, lr=0.1, thread=8)
# 保存训练之后的模型
# model.save_model("fil9.bin")


# 使用fasttext.load_model加载模型
model = fasttext.load_model("fil9.bin")
print(model.get_word_vector("the"))

# 查找和音乐相近的词
print(model.get_nearest_neighbors('music'))
