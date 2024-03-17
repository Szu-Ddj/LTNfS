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


class BERT_FOL_R(nn.Module):
    def __init__(self, bert, opt,tokenizer):
        super(BERT_FOL_R, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense1 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.tokenizer = tokenizer
    def forward(self, inputs):
        fol_bert_indices, fol_bert_type,fol_bert_mask,fol_pred_bert_indices, fol_pred_bert_type,fol_pred_bert_mask = inputs
        # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]
        hidden_state,pooled_output= self.bert(fol_bert_indices, token_type_ids=fol_bert_type,attention_mask=fol_bert_mask,return_dict=False)
        pred_hidden_state,pred_pooled_output= self.bert(fol_pred_bert_indices, token_type_ids=fol_pred_bert_type,attention_mask=fol_pred_bert_mask,return_dict=False)

        out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
        for index in range(hidden_state.size(0)):
            out[index] = evaluate_expression(fol_bert_indices[index],hidden_state[index]).squeeze()
        # print(out.size(),pred_pooled_output.size())
        out = torch.cat((out,0.2*pred_pooled_output),-1)
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
        # return pooled_output
# # -*- coding: utf-8 -*-
# # file: BERT_SPC.py
# # author: songyouwei <youwei0314@gmail.com>
# # Copyright (C) 2019. All Rights Reserved.
# import torch
# import torch.nn as nn
# '''
# ∀ 1
# ∃ 2
# ( 1006
# ) 1007
# ¬ 1078
# ∨ 1603
# ∧ 1602
# → 1585
# '''
# def evaluate_expression(expression,states):
#     def process_expression(expr,sts):
#         stack = torch.Tensor().cuda()
#         stacksts = torch.Tensor().cuda()
#         i = 0
#         result = states
#         while i < len(expr):
#             if expr[i] == 1006:
#                 stack = torch.concat((stack,torch.tensor([1006]).cuda()))
#             elif expr[i] == 1007:
#                 sub_expr = torch.Tensor().cuda()
#                 sub_sts = torch.Tensor().cuda()
#                 # print(stack,expr,len(sts),len(expr))
#                 while stack[-1].item() != 1006:
#                     sub_expr = torch.concat((stack[-1:],sub_expr),0)
#                     sub_sts = torch.concat((stacksts[-1:],sub_sts),0)
#                     stack = stack[:-1]
#                     stacksts = stacksts[:-1]
#                 stack = stack[:-1]
#                 result,rs = evaluate_sub_expression(sub_expr,sub_sts)
#                 stack = torch.concat((stack,rs),0)
#                 stacksts = torch.concat((stacksts,result),0)
#             else:
#                 stack = torch.concat((stack,expr[i:i+1]),0)
#                 stacksts = torch.concat((stacksts,sts[i:i+1]),0)
#             i += 1
#         # print('xx',result.size())
#         # print('xx',)
#         # result = evaluate_sub_expression(sub_expr,sub_sts)
#         result = torch.mean(result,keepdim=True,dim=0)
#         return result

#     def evaluate_sub_expression(expr,sts):
#         opera_list = [1603,1602,1585]
#         prei = -1
#         result = sts
#         nosigns = False
#         for _index in range(expr.size(0)):
#             _eitem = expr[_index]
#             _item = sts[_index]
#             if _eitem == 1078:
#                 nosigns = True
#             if _eitem in opera_list:
#                 if nosigns:
#                     if expr[prei+1:_index][0] != 1078:
#                         print(expr)
#                         print(expr[prei+1:_index])
#                     value = 1 - torch.mean(sts[prei+2:_index],0,keepdim=True)
#                     nosigns = False
#                 else:
#                     value = torch.mean(sts[prei+1:_index],0,keepdim=True)

#                 if prei != -1:

#                     value = operate(prev,value,expr[prei])
#                 prev = value
#                 prei = _index

#             if _index == len(expr) - 1:
#                 if prei == -1:

#                     break
#                 # print(prev,sts[prei+1:len(expr)],expr[prei])
#                 result = operate(prev,torch.mean(sts[prei+1:len(expr)],0,keepdim=True),expr[prei])

#         return result,torch.ones(result.size(0)).cuda()*-1
#     def operate(left,right,signs):
#         # if 1078 in left:
#         #     left = 1 - torch.mean(left[1:,:],0,keepdim=True)
#         # else:
#         # left = torch.mean(left,0,keepdim=True)

#         # if 1078 in right:
#         #     right = 1 - torch.mean(right[1:,:],0,keepdim=True)
#         # else:
#         # right = torch.mean(right,0,keepdim=True)
            
            
#         if signs == 1602:
#             return torch.mul(left,right)
#         if signs == 1603:
#             return left + right - torch.mul(left,right)
#         if signs == 1585:
#             return 1 - left + torch.mul(left,right)

        

#     return process_expression(expression,states)
# class BERT_FOL(nn.Module):
#     def __init__(self, bert, opt):
#         super(BERT_FOL, self).__init__()
#         self.bert = bert
#         self.dropout = nn.Dropout(opt.dropout)
#         self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)

#     def forward(self, inputs):
#         text_bert_indices, text_bert_type,text_bert_mask = inputs[0], inputs[1], inputs[2]
#         # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]
#         hidden_state,pooled_output= self.bert(text_bert_indices, token_type_ids=text_bert_type,attention_mask=text_bert_mask,return_dict=False)

#         out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
#         for index in range(hidden_state.size(0)):
#             out[index] = evaluate_expression(text_bert_indices[index],hidden_state[index]).squeeze()

#         out = self.dropout(out)
#         logits = self.dense(out)
#         # print(logits.size(), logits.device)
#         return logits
#         # return pooled_output
