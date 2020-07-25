#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:29:25 2020

@author: safak
"""

import torch 
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
from src import models
from src import process
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", "-d", help="set data")
parser.add_argument("--epochs", "-e", help="set number of epochs")
args = parser.parse_args()

CUDA = torch.cuda.is_available()
EMBED_SIZE = 128
HIDDEN_SIZE = 1024
NUM_LAYERS = 1
EPOCHS = args.epochs
BATCH_SIZE = 16
TIMESTEPS = 30
LEARNING_RATE = 0.001

corpus = process.NLP()
index_of_words = corpus.tokenize(args.datapath,BATCH_SIZE)

VOCAB_SIZE = len(corpus.dictionary)
NUMBER_OF_BATCHES = index_of_words.shape[1] // TIMESTEPS

model = models.Author(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, BATCH_SIZE)
if CUDA:
    model = model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

for epoch in range(int(EPOCHS)):
    #h_t and c_t
    states = (torch.zeros(NUM_LAYERS,BATCH_SIZE, HIDDEN_SIZE,device = "cuda"), 
              torch.zeros(NUM_LAYERS,BATCH_SIZE, HIDDEN_SIZE,device="cuda"))
    for i in range(0,index_of_words.size(1) - TIMESTEPS, TIMESTEPS):
        inputs = index_of_words[:, i:i+TIMESTEPS]
        targets = index_of_words[:,(i+1):(i+1)+TIMESTEPS]
        if CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs, hc = model.forward(inputs,states)
        cost = loss(outputs,targets.reshape(-1))
        model.zero_grad()
        cost.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        step = (i+1) // TIMESTEPS
        if(step % 100 == 0):
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1,EPOCHS,cost.item()))   
            
torch.save(model.state_dict(), "./checkpoints/model.pth")
print("Model is trained and saved at ./checkpoints/model.pth")
