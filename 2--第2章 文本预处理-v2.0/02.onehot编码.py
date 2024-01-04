# 导入用于对象保存与加载的joblib
import joblib
# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer

# 假定vocab为语料集所有不同词汇集合
vocab = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
# 实例化一个词汇映射器对象
t = Tokenizer(num_words=None, char_level=False)
# 使用映射器拟合现有文本数据
t.fit_on_texts(vocab)

for token in vocab:
    zero_list = [0] * len(vocab)
    # 使用映射器转化现有文本数据, 每个词汇对应从1开始的自然数
    # 返回样式如: [[2]], 取出其中的数字需要使用[0][0]
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为:", zero_list)

# 使用joblib工具保存映射器, 以便之后使用
tokenizer_path = "./Tokenizer"
print(joblib.dump(t, tokenizer_path))

# 比如我有一个此表，里面有3w个词汇。那么每次表示一个词那么都是3w元素的一个列表，然后对应的词汇哪个索引是1别的都是0.这样就可以表示这个词的位置了。
# 这个的好处就是把词用一个向量表示了方便计算进行科学计算。
# 缺点就是每个向量的长度过大，占据大量内存.比如一共10w词，那么表示每一个词都是10w的长度的列表

# 加载之前保存的Tokenizer, 实例化一个t对象
t = joblib.load(tokenizer_path)
# 编码token为"李宗盛"
token = "李宗盛"
# 使用t获得token_index
token_index = t.texts_to_sequences([token])[0][0] - 1
# 初始化一个zero_list
zero_list = [0] * len(vocab)
# 令zero_List的对应索引为1
zero_list[token_index] = 1
print(token, "的one-hot编码为:", zero_list)
