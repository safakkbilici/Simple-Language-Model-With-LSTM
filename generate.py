import torch 
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
from src import process
from src import models
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", "-d", help="set data")
parser.add_argument("--textname", "-t", help="set new text file name")
parser.add_argument("--nwords", "-n", help="set number of generated words")
args = parser.parse_args()


CUDA = torch.cuda.is_available()
EMBED_SIZE = 128
HIDDEN_SIZE = 1024
NUM_LAYERS = 1
EPOCHS = 30
BATCH_SIZE = 16
corpus = process.NLP()
index_of_words = corpus.tokenize(args.datapath,BATCH_SIZE)
VOCAB_SIZE = len(corpus.dictionary)
model = models.Author(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, BATCH_SIZE)
model.load_state_dict(torch.load("./checkpoints/model.pth"))
""" 
MODEL SUMMARY AND NUMBER OF TRAINABLE PARAMETERS
-------------------------------------------------
print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
"""
with torch.no_grad():
        with open(args.textname, 'w') as f:
            states = (torch.zeros(NUM_LAYERS,1, HIDDEN_SIZE), 
                      torch.zeros(NUM_LAYERS,1, HIDDEN_SIZE))
            input = torch.randint(0, VOCAB_SIZE, (1,)).long().unsqueeze(1)
            for i in range(int(args.nwords)):
                output, hc = model.forward(input,states)
                probabilities = output.exp()
                word_id = torch.multinomial(probabilities, num_samples = 1).item()
                print(word_id)
                input.fill_(word_id)
                
                word = corpus.dictionary.index2word[word_id]
                word = '\n' if word == '<EOS>' else word +' '
                f.write(word)
print("New text is ready.")
