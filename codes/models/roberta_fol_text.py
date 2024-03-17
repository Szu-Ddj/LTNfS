# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
'''
∀ 1
∃ 2
( 1006
) 1007
¬ 1078
∨ 1603
∧ 1602
→ 1585
'''
from layers.operate_bart import evaluate_expression


class ROBERTA_FOL_T(nn.Module):
    def __init__(self, bert, opt,tokenizer):
        super(ROBERTA_FOL_T, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense1 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.tokenizer = tokenizer
    def forward(self, inputs):
        fol_bert_indices,fol_bert_mask,fol_pred_bert_indices,fol_pred_bert_mask,ti,tm,mlm = inputs
        # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]

        hidden_state,pooled_output= self.bert(fol_bert_indices,attention_mask=fol_bert_mask,return_dict=False)
        # pred_hidden_state,pred_pooled_output= self.bert(fol_pred_bert_indices,attention_mask=fol_pred_bert_mask,return_dict=False)
        text_hidden_state,text_pooled_output= self.bert(ti,attention_mask=tm,return_dict=False)
        
        #
        # print(mlm,mlm.size())
        # print(text_hidden_state.size(),text_hidden_state[mlm >= 0].size())
        # print(text_hidden_state[mlm >= 0].view(hidden_state.size(0),1,hidden_state.size(-1)).size())
        text_hidden_state = text_hidden_state[mlm >= 0].view(text_hidden_state.size(0),1,text_hidden_state.size(-1)).squeeze(1)
        # print(text_hidden_state.size())
        #
        out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
        for index in range(hidden_state.size(0)):
            out[index] = evaluate_expression(fol_bert_indices[index],hidden_state[index],1,1,1).squeeze()
        # print(out.size())
        out = torch.cat((out,0.7*text_hidden_state),-1)
        out = self.dropout(out)
        out = self.dense1(out)
        self.tokenizer.get_labels()
        logits = [
            torch.mm(
                out[:,:],
                self.bert.embeddings.word_embeddings.weight[i].transpose(1,0)
            ) for i in self.tokenizer.prompt_label_idx
        ]
        # logits = self.dense(out)
        # print(logits.size(), logits.device)
        return logits