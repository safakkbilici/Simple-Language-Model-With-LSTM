import torch 
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
class Dictionary(object):
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.index = 0
        
    def add_word(self, word):
        if(word not in self.word2index):
            self.word2index[word] = self.index
            self.index2word[self.index] = word
            self.index = self.index + 1
    
    def __len__(self):
        return len(self.word2index)

class NLP(object):
    def __init__(self):
        self.dictionary = Dictionary()
    
    def tokenize(self, data_path, batch_size=20):
        with open(data_path,'r') as f:
            tokens=0
            for line in f:
                string_array = line.split() + ['<EOS>']
                tokens = tokens + len(string_array)
                for a_word in string_array:
                    self.dictionary.add_word(a_word)
        index_of_words = torch.LongTensor(tokens)
        index = 0
        with open(data_path,'r') as f:
            for line in f:
                string_array = line.split() + ['<EOS>']
                for a_word in string_array:
                    index_of_words[index] = self.dictionary.word2index[a_word]
                    index = index + 1
        #print(index_of_words.shape)
        number_of_batches = index_of_words.shape[0] // batch_size
        index_of_words = index_of_words[:number_of_batches*batch_size]
        index_of_words = index_of_words.view(batch_size,-1)
        return index_of_words
