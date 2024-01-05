# 导入工具包
import torch
import torch.nn as nn
import torch.nn.functional as F

# '''
# 第一个参数：input_size(输入张量x的维度)
# 第二个参数：hidden_size(隐藏层的维度， 隐藏层的神经元个数)
# 第三个参数：num_layer(隐藏层的数量)
# '''
rnn = nn.RNN(5, 6, 3)  # A
# '''
# 第一个参数：sequence_length(输入序列的长度)
# 第二个参数：batch_size(批次的样本数量)
# 第三个参数：input_size(输入张量的维度)
# '''
input = torch.randn(2, 3, 5)  # B
# '''
# 第一个参数：num_layer * num_directions(层数*网络方向)
# 第二个参数：batch_size(批次的样本数)
# 第三个参数：hidden_size(隐藏层的维度， 隐藏层神经元的个数)
# '''
h0 = torch.randn(3, 3, 6)  # C
output, hn = rnn(input, h0)
print(output)
print('outputshape==', output.shape)  # torch.Size([1, 3, 6])  1--》 seq_length 3 --batch_size 6 -- 隐藏层节点
print(hn)
print('hnshape===', hn.shape)  # torch.Size([1, 3, 6])
print('nn.shape===', rnn)

# 注意点： 上面中的三个一的意义不一样 A和C的一需要保持一致， 也可以自己指定
# B中的1可以自己任意改写， 因为是程序员自己指定，
