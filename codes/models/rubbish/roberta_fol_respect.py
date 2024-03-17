# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
# from transformers import BertModel, BartConfig, BartForSequenceClassification
from layers.operate_bart import evaluate_expression

class ROBERTA_FOL_R(nn.Module):
    def __init__(self, bert, opt,tokenizer):
        super(ROBERTA_FOL_R, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense1 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.tokenizer = tokenizer


        self.notnetw = nn.Sequential(
            nn.LeakyReLU(negative_slope=-0.2),
            nn.Linear(self.bert.config.hidden_size,self.bert.config.hidden_size),
                                    )
        self.andnetw = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size,1),
                                    )
        self.ornetw = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size*2,self.bert.config.hidden_size),
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
        fol_bert_indices,fol_bert_mask,fol_pred_bert_indices,fol_pred_bert_mask = inputs
        # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]
        hidden_state,pooled_output= self.bert(fol_bert_indices, attention_mask=fol_bert_mask,return_dict=False)
        pred_hidden_state,pred_pooled_output= self.bert(fol_pred_bert_indices, attention_mask=fol_pred_bert_mask,return_dict=False)

        out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
        for index in range(hidden_state.size(0)):
            out[index] = evaluate_expression(fol_bert_indices[index],hidden_state[index],self.andnet,self.ornet,self.notnet).squeeze()
        # print(out.size(),pred_pooled_output.size())
        out = torch.cat((out,0.2*pred_pooled_output),-1)
        out = self.dropout(out)
        out = self.dense1(out)
        logits = self.dense(out)
        self.tokenizer.get_labels()
        logits = [
            torch.mm(
                out[:,:],
                self.bert.embeddings.word_embeddings.weight[i].transpose(1,0)
            ) for i in self.tokenizer.prompt_label_idx
        ]

        return logits

