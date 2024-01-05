"""
我们观察事物时，之所以能够快速判断一种事物(当然允许判断是错误的),
是因为我们大脑能够很快把注意力放在事物最具有辨识度的部分从而作出判断，而并非是从头到尾的观察一遍事物后，
才能有判断结果. 正是基于这样的理论，就产生了注意力机制

它需要三个指定的输入Q(query), K(key), V(value), 然后通过计算公式得到注意力的结果,
这个结果代表query在key和value作用下的注意力表示. 当输入的Q=K=V时, 称作自注意力计算规则.

注意力机制实现步骤
第一步: 根据注意力计算规则, 对Q，K，V进行相应的计算.
第二步: 根据第一步采用的计算方法, 如果是拼接方法，则需要将Q与第二步的计算结果再进行拼接, 如果是转置点积, 一般是自注意力, Q与V相同, 则不需要进行与Q的拼接.
第三步: 最后为了使整个attention机制按照指定尺寸输出, 使用线性层作用在第二步的结果上做一个线性变换, 得到最终对Q的注意力表示.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        """初始化函数中的参数有5个, query_size代表query的最后一维大小
           key_size代表key的最后一维大小, value_size1代表value的导数第二维大小,
           value = (1, value_size1, value_size2)
           value_size2代表value的倒数第一维大小, output_size输出的最后一维大小"""
        super(Attn, self).__init__()
        # 将以下参数传入类中
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 初始化注意力机制实现第一步中需要的线性层.
        self.attn = nn.Linear(self.query_size + self.key_size, value_size1)

        # 初始化注意力机制实现第三步中需要的线性层.
        self.attn_combine = nn.Linear(self.query_size + value_size2, output_size)

    def forward(self, Q, K, V):
        """forward函数的输入参数有三个, 分别是Q, K, V, 根据模型训练常识, 输入给Attion机制的
           张量一般情况都是三维张量, 因此这里也假设Q, K, V都是三维张量"""

        # 第一步, 按照计算规则进行计算,
        # 我们采用常见的第一种计算规则
        # 将Q，K进行纵轴拼接, 做一次线性变化, 最后使用softmax处理获得结果
        attn_weights = F.softmax(
            self.attn(torch.cat((Q[0], K[0]), 1)), dim=1)

        # 然后进行第一步的后半部分, 将得到的权重矩阵与V做矩阵乘法计算,
        # 当二者都是三维张量且第一维代表为batch条数时, 则做bmm运算
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)

        # 之后进行第二步, 通过取[0]是用来降维, 根据第一步采用的计算方法,
        # 需要将Q与第一步的计算结果再进行拼接
        output = torch.cat((Q[0], attn_applied[0]), 1)

        # 最后是第三步, 使用线性层作用在第三步的结果上做一个线性变换并扩展维度，得到输出
        # 因为要保证输出也是三维张量, 因此使用unsqueeze(0)扩展维度
        output = self.attn_combine(output).unsqueeze(0)
        return output, attn_weights


if __name__ == '__main__':
    query_size = 32
    key_size = 32
    value_size1 = 32
    value_size2 = 64
    output_size = 64
    attn = Attn(query_size, key_size, value_size1, value_size2, output_size)
    Q = torch.randn(1, 1, 32)
    K = torch.randn(1, 1, 32)
    V = torch.randn(1, 32, 64)
    out = attn(Q, K, V)
    print(out[0])
    print(out[1])
