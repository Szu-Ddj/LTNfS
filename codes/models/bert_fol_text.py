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
from layers.opera_best_bert import evaluate_expression


class BERT_FOL_T(nn.Module):
    def __init__(self, bert, opt,tokenizer):
        super(BERT_FOL_T, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense1 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.dense2 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.tokenizer = tokenizer
        self.with_text = opt.with_text
    def forward(self, inputs):
        # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]


        fol_bert_indices, fol_bert_type,fol_bert_mask,ti,tt,tm,mlm = inputs
        hidden_state,pooled_output= self.bert(fol_bert_indices, token_type_ids=fol_bert_type,attention_mask=fol_bert_mask,return_dict=False)
        if self.with_text:
            text_hidden_state,pooled_output= self.bert(ti, token_type_ids=tt,attention_mask=tm,return_dict=False)
            text_hidden_state = text_hidden_state[mlm >= 0].view(hidden_state.size(0),1,hidden_state.size(-1)).squeeze(1)
        # else:
        #     fol_bert_indices, fol_bert_type,fol_bert_mask = inputs

        # pred_hidden_state,pred_pooled_output= self.bert(fol_pred_bert_indices, token_type_ids=fol_pred_bert_type,attention_mask=fol_pred_bert_mask,return_dict=False)
        
        # print(mlm,mlm.size())
        # print(text_hidden_state.size(),text_hidden_state[mlm >= 0].size())
        # print(text_hidden_state[mlm >= 0].view(hidden_state.size(0),1,hidden_state.size(-1)).size())
        # print(text_hidden_state.size())
        out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
        for index in range(hidden_state.size(0)):
            out[index] = evaluate_expression(fol_bert_indices[index],hidden_state[index]).squeeze()
        # print(out.size())
        
        if self.with_text:
            out = torch.cat((out,text_hidden_state),-1)
            out = self.dropout(out)
            out = self.dense1(out)
        else:
            out = self.dropout(out)
            out = self.dense2(out)

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