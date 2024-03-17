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

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # print(adj.size(), hidden.size())
        # assert False        
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class BERT_FOL_G(nn.Module):
    def __init__(self, bert, opt,tokenizer):
        super(BERT_FOL_G, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense1 = nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size)
        self.dense2 = nn.Linear(self.bert.config.hidden_size*3, self.bert.config.hidden_size)
        # self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.graph_dense11 = nn.Linear(20, 1)
        self.graph_dense12 = nn.Linear(20, 1)
        self.graph_dense13 = nn.Linear(20, 1)
        # self.graph_dense2 = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.gcn = GraphConvolution(self.bert.config.hidden_size,self.bert.config.hidden_size)
        self.tokenizer = tokenizer
        self.with_text = opt.with_text

    def forward(self, inputs):
        # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]


        fol_bert_indices, fol_bert_type,fol_bert_mask,ti,tt,tm,mlm,graph_matrixs,graph_texts = inputs
        
        
        #graph
        # print(graph_matrixs)
        graph_matrixs = graph_matrixs.transpose(0,1)
        # print(graph_matrixs)
        # assert False
        graph_texts = graph_texts.transpose(0,1)
        try:
            assert graph_texts.size(0) == 3 == graph_matrixs.size(0)
        except:
            print(graph_texts.size(0), graph_matrixs.size(0))
            assert graph_texts.size(0) == 3 == graph_matrixs.size(0)
        for _index,(graph_matrix,graph_text) in enumerate(zip(graph_matrixs,graph_texts)):
            # print(graph_matrix.size(),graph_text.size())
            graph_hidden_state,_= self.bert(graph_text,return_dict=False)
            graph_text = self.gcn(graph_hidden_state,graph_matrix)
            # graph_text = self.gcn(graph_text,graph_matrix)
            if _index == 0:
                _graph_out = self.graph_dense11(graph_text.transpose(1,2)).squeeze(-1)
            elif _index == 1:
                _graph_out = self.graph_dense12(graph_text.transpose(1,2)).squeeze(-1)
            elif _index == 2:
                _graph_out = self.graph_dense13(graph_text.transpose(1,2)).squeeze(-1)
            if _index == 0:
                graph_out = _graph_out
            else:
                graph_out = torch.concat((graph_out,_graph_out),-1)
        
        # hidden_state,pooled_output= self.bert(fol_bert_indices, token_type_ids=fol_bert_type,attention_mask=fol_bert_mask,return_dict=False)
        if self.with_text:
            text_hidden_state,pooled_output= self.bert(ti, token_type_ids=tt,attention_mask=tm,return_dict=False)
            text_hidden_state = text_hidden_state[mlm >= 0].view(text_hidden_state.size(0),1,text_hidden_state.size(-1)).squeeze(1)

        # out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
        # for index in range(hidden_state.size(0)):
            # out[index] = evaluate_expression(fol_bert_indices[index],hidden_state[index]).squeeze()

        
        if self.with_text:
            out = torch.cat((graph_out,text_hidden_state),-1)
            out = self.dropout(out)
            out = self.dense1(out)
        else:
            out = self.dropout(graph_out)
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