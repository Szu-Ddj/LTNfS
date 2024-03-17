# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
from layers.opera_best import evaluate_expression
'''
( 1
) 2
¬ 3
∨ 4
∧ 5
→ 6
'''
class LSTM_FOL(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM_FOL, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,bidirectional=True,dropout=opt.dropout)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)
        self.notnetw = nn.Sequential(
            nn.LeakyReLU(negative_slope=-0.2),
            nn.Linear(opt.hidden_dim*2,opt.hidden_dim*2),
                                    )
        self.andnetw = nn.Sequential(
            nn.Linear(opt.hidden_dim*2,1),
                                    )
        self.ornetw = nn.Sequential(
            nn.Linear(opt.hidden_dim*4,opt.hidden_dim*2),
                                    )

    def notnet(self,x):
        # print(type(self.notnetw(x)),self.notnetw(x).size())
        return self.notnetw(x)
    def andnet(self,x,y):
        # print(self.andnetw(torch.cat((x,y),-2)).size())
        return torch.matmul(torch.cat((x,y),-2).transpose(-1,-2),self.andnetw(torch.cat((x,y),-2))).transpose(-1,-2)
    def ornet(self,x,y):
        # print(self.ornetw(torch.cat((x,y),-1)).size())
        return self.ornetw(torch.cat((x,y),-1))
    
    def forward(self, inputs):
        # self.embed = nn.Embedding()
        fol_indices = inputs[0]
        x = self.embed(fol_indices)
        # print("x:",x)
        x_len = torch.sum(fol_indices != 0, dim=-1)
        out_states, (h_n, _) = self.lstm(x, x_len)
        # print(out_states.size(),fol_indices.size())
        # print(fol_indices[:,:22])
        out = torch.zeros([out_states.size(0),out_states.size(-1)]).cuda()
        for index in range(fol_indices.size(0)):
            out[index] = evaluate_expression(fol_indices[index,:22],out_states[index],self.andnet,self.ornet,self.notnet)
        # print(out_states.size())
        # print(out.size(),h_n[0].size())
        # print(out.size(),h_n.size())
        # assert False
        out = self.dense(out)
        return out
