import torch 
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
class Author(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers,batch_size):
        super(Author,self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first = True)
        self.linear = nn.Linear(hidden_size,vocab_size)
    def forward(self,x,h):
        x = self.embed(x)
        out, (h,c) = self.lstm(x,h) # samples x timesteps x output_features
        out = out.reshape(out.size(0)*out.size(1),out.size(2))
        out = self.linear(out)
        return out, (h,c)
