import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        # 必须项,让父类进行初始化操作
        super(RNN,self).__init__() 
        # 参数初始化操作,用来接受全局参数使用
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_szie = output_size
        self.num_layers = num_layers

        # 实例化RNN
        self.rnn = nn.RNN(input_size,hidden_size,num_layers)
        # 实例化全链接线形层,作用是把RNN输出为度转换成指定的维度
        self.liner = nn.Linear(hidden_size, output_size)
        # 实例化 softmax层
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input1, hidden):
        # 先进行维度变换操作把[1,n]变成[1,1,n]方便进行张量计算
        input1 = input1.unsqueeze(0)
        # 然后再把输入的张量输入到RNN对象中
        rr,hn = self.rnn(input1,hidden)
        # 将RNN获取的结果进行线性变化和softmax的处理
        return self.softmax(self.liner(rr)),hn

    def initHidden(self):
        return torch.zeros(self.num_layers,1,self.hidden_size)




