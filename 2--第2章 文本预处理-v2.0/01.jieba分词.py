import jieba

content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
res = jieba.cut(content, cut_all=False)  # cut_all默认为False
# 这里返回的是一个迭代对象
print(res)

# 若需直接返回列表内容, 使用jieba.lcut即可
res = jieba.lcut(content, cut_all=False)
print(res)

# 全模式分词
# 把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能消除歧义.
res = jieba.lcut(content, cut_all=True)
print(res)

# 搜索引擎模式分词,这种模式感觉更好一点
# 在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
res = jieba.lcut_for_search(content)
print(res)

# jieba分词还支持繁体
content = "煩惱即是菩提，我暫且不提"
print(jieba.lcut(content))
