import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    result = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return result

def attention(x):
    n, d = x.shape
    Wq = np.random.rand(d, d)
    Wk = np.random.rand(d, d)
    Wv = np.random.rand(d, d)
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    A = q @ k.T
    A = A / np.sqrt(d)
    A_hat = softmax(A)
    output = A_hat @ v
    print(output.shape) # n, d


def multi_head_attention(x, head_n=16):
    n, d = x.shape
    assert d % head_n == 0
    Wq = np.random.rand(d, d)
    Wk = np.random.rand(d, d)
    Wv = np.random.rand(d, d)
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    q = np.reshape(q, (n, head_n, d // head_n))
    k = np.reshape(k, (n, head_n, d // head_n))
    v = np.reshape(v, (n, head_n, d // head_n))
    q = np.transpose(q, (1, 0, 2))  # head_n, n, d // head_n
    k = np.transpose(k, (1, 0, 2))
    v = np.transpose(v, (1, 0, 2))
    A = q @ np.transpose(k, (0, 2, 1))
    A = A / np.sqrt(d // head_n)
    A_hat = softmax(A) # head_n, n, n
    output = A_hat @ v # head_n, n, d // head_n
    output = np.transpose(output, (1, 0, 2))    # n, head_n, d // head_n
    output = np.reshape(output, (n, d)) 
    print(output.shape) # n, d
    

if __name__ == "__main__":
    attention(np.random.rand(512, 768))
    multi_head_attention(np.random.rand(512, 768))


import torch
import torch.nn as nn

from math import sqrt

class SelfAttention(nn.Module):
    def __init__(self,input_dim:int,dim_k,dim_v):
        super(SelfAttention,self).__init__()
        self.q_linear = nn.Linear(input_dim,dim_k)
        self.k_linear = nn.Linear(input_dim,dim_k)
        self.v_linear = nn.Linear(input_dim,dim_v)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        attention = torch.matmul(Q,K.transpose(1,2))
        attention = attention / sqrt(K.size(-1))
        attention = self.softmax(attention)
        out = torch.matmul(attention,V)

        return out