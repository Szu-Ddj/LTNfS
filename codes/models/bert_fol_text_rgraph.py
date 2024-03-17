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
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data, Batch
from torch_geometric import data,loader
class RGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations=4):
        super(RGCN, self).__init__()
        # RGCN卷积层
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type):
        # 通过两个RGCN层传递数据
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x


class BERT_FOL_R(nn.Module):
    def __init__(self, bert, opt,tokenizer):
        super(BERT_FOL_R, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense1 = nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size)
        self.dense2 = nn.Linear(self.bert.config.hidden_size*3, self.bert.config.hidden_size)
        # self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.graph_dense11 = nn.Linear(opt.nodes_num, 1)
        self.graph_dense12 = nn.Linear(opt.nodes_num, 1)
        self.graph_dense13 = nn.Linear(opt.nodes_num, 1)
        # self.graph_dense2 = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.gcn = RGCN(self.bert.config.hidden_size,self.bert.config.hidden_size)
        self.tokenizer = tokenizer
        self.with_text = opt.with_text
        self.batch_size = opt.batch_size

    def forward(self, inputs):
        # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]


        fol_bert_indices, fol_bert_type,fol_bert_mask,ti,tt,tm,mlm,graph_indexs,graph_types,graph_texts = inputs
        
        
        #graph
        # print(graph_matrixs)
        graph_indexs = graph_indexs.transpose(0,1)
        graph_types = graph_types.transpose(0,1)
        # print(graph_matrixs)
        # assert False
        graph_texts = graph_texts.transpose(0,1)
        try:
            assert graph_texts.size(0) == 3 == graph_indexs.size(0) == graph_types.size(0)
        except:
            print(graph_texts.size(0), graph_indexs.size(0),graph_types.size(0))
            assert graph_texts.size(0) == 3 == graph_indexs.size(0) == graph_types.size(0)
        for _index,(graph_index,graph_type,graph_text) in enumerate(zip(graph_indexs,graph_types,graph_texts)):
            graph_hidden_state,_= self.bert(graph_text,return_dict=False)
            # print(graph_hidden_state.size(),graph_index.size(),graph_type.size())
            # for sgraph_hidden_state,sgraph_index,sgraph_type in zip(graph_hidden_state,graph_index,graph_type):
            #     print(sgraph_hidden_state.size(), sgraph_index[:,:len(sgraph_type[sgraph_type >= 0])].size(), sgraph_type[sgraph_type >= 0].size())
            
            data_list = [
                Data(x=sgraph_hidden_state, edge_index=sgraph_index[:,:len(sgraph_type[sgraph_type >= 0])], edge_type=sgraph_type[sgraph_type >= 0]) for sgraph_hidden_state,sgraph_index,sgraph_type in zip(graph_hidden_state,graph_index,graph_type)
            ]
            batch_data = list(loader.DataLoader(data_list, batch_size=self.batch_size,shuffle=False))[0]
            # print(batch_data.x.size(), batch_data.edge_index.size(), batch_data.edge_type.size())
            graph_text_out = self.gcn(batch_data.x, batch_data.edge_index, batch_data.edge_type)
            
            # x_batch = torch.cat([single_graph_text for single_graph_text in graph_hidden_state],dim=0)
            # # print(graph_index.size(),graph_type.size())
            # # for single_graph_index,single_graph_type in zip(graph_index,graph_type):
            # #     print(single_graph_index.size(),single_graph_type.size())
            #     # print(single_graph_index[single_graph_type>=0])
            # edge_index_batch = torch.cat([single_graph_index[single_graph_type>=0] for single_graph_index,single_graph_type in zip(graph_index,graph_type)],dim=0)
            # edge_type_batch = torch.cat([single_graph_type[single_graph_type>=0] for single_graph_type in graph_type],dim=0)
            # gbatch = torch.cat([torch.full_like(single_graph_text[:, 0], i) for i, single_graph_text in enumerate(graph_hidden_state)],dim=0)
            # print(edge_type_batch.size(),edge_index_batch.size(),x_batch.size(),gbatch.size())
            # graph_text_out = self.gcn(x_batch, edge_index_batch, edge_type_batch, gbatch)
            graph_output_split = torch.split(graph_text_out, [len(d) for d in graph_hidden_state], dim=0)
            graph_text_out = torch.cat([_item.unsqueeze(0) for _item in graph_output_split],0)

            # graph_text_out = self.gcn(graph_text_out, edge_index_batch, edge_type_batch, batch)
            if _index == 0:
                _graph_out = self.graph_dense11(graph_text_out.transpose(1,2)).squeeze(-1)
            elif _index == 1:
                _graph_out = self.graph_dense12(graph_text_out.transpose(1,2)).squeeze(-1)
            elif _index == 2:
                _graph_out = self.graph_dense13(graph_text_out.transpose(1,2)).squeeze(-1)
            
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
# from layers.opera_best_bert import evaluate_expression
# import torch
# import torch.nn as nn
# from torch_geometric.nn import RGCNConv

# class RGCN(nn.Module):
#     def __init__(self, in_channels, out_channels, num_relations=4):
#         super(RGCN, self).__init__()
#         # RGCN卷积层
#         self.conv1 = RGCNConv(in_channels, out_channels, num_relations=num_relations)
#         self.conv2 = RGCNConv(out_channels, out_channels, num_relations=num_relations)

#     def forward(self, x, edge_index, edge_type,batch=None):
#         # 通过两个RGCN层传递数据
#         x = self.conv1(x, edge_index, edge_type)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index, edge_type)
#         return x


# class BERT_FOL_R(nn.Module):
#     def __init__(self, bert, opt,tokenizer):
#         super(BERT_FOL_R, self).__init__()
#         self.bert = bert
#         self.dropout = nn.Dropout(opt.dropout)
#         self.dense1 = nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size)
#         self.dense2 = nn.Linear(self.bert.config.hidden_size*3, self.bert.config.hidden_size)
#         # self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
#         self.graph_dense11 = nn.Linear(20, 1)
#         self.graph_dense12 = nn.Linear(20, 1)
#         self.graph_dense13 = nn.Linear(20, 1)
#         # self.graph_dense2 = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
#         self.gcn = RGCN(self.bert.config.hidden_size,self.bert.config.hidden_size)
#         self.tokenizer = tokenizer
#         self.with_text = opt.with_text

#     def forward(self, inputs):
#         # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]


#         fol_bert_indices, fol_bert_type,fol_bert_mask,ti,tt,tm,mlm,graph_indexs,graph_types,graph_texts = inputs
        
        
#         #graph
#         # print(graph_matrixs)
#         graph_indexs = graph_indexs.transpose(0,1)
#         graph_types = graph_types.transpose(0,1)
#         # print(graph_matrixs)
#         # assert False
#         graph_texts = graph_texts.transpose(0,1)
#         try:
#             assert graph_texts.size(0) == 3 == graph_indexs.size(0) == graph_types.size(0)
#         except:
#             print(graph_texts.size(0), graph_indexs.size(0),graph_types.size(0))
#             assert graph_texts.size(0) == 3 == graph_indexs.size(0) == graph_types.size(0)
#         for _index,(graph_index,graph_type,graph_text) in enumerate(zip(graph_indexs,graph_types,graph_texts)):
#             graph_hidden_state,_= self.bert(graph_text,return_dict=False)

#             # x_batch = torch.cat([single_graph_text for single_graph_text in graph_hidden_state],dim=0)
#             # # print(graph_index.size(),graph_type.size())
#             # # for single_graph_index,single_graph_type in zip(graph_index,graph_type):
#             # #     print(single_graph_index.size(),single_graph_type.size())
#             #     # print(single_graph_index[single_graph_type>=0])
#             # edge_index_batch = torch.cat([single_graph_index[single_graph_type>=0] for single_graph_index,single_graph_type in zip(graph_index,graph_type)],dim=0)
#             # edge_type_batch = torch.cat([single_graph_type[single_graph_type>=0] for single_graph_type in graph_type],dim=0)
#             # gbatch = torch.cat([torch.full_like(single_graph_text[:, 0], i) for i, single_graph_text in enumerate(graph_hidden_state)],dim=0)
#             # print(edge_type_batch.size(),edge_index_batch.size(),x_batch.size(),gbatch.size())
#             # graph_text_out = self.gcn(x_batch, edge_index_batch, edge_type_batch, gbatch)
#             # graph_output_split = torch.split(graph_text_out, [len(d) for d in graph_hidden_state], dim=0)
#             # graph_text_out = torch.cat([_item.unsqueeze(0) for _item in graph_output_split],0)
#             graph_text_out = []
#             for sgraph_index,sgraph_type,sgraph_text in zip((graph_index,graph_type,graph_text)):
#                 graph_text_out.append(self.gcn(sgraph_index,sgraph_type,sgraph_text))
#             graph_text_out = torch.cat()


#             # graph_text_out = self.gcn(graph_text_out, edge_index_batch, edge_type_batch, batch)
#             if _index == 0:
#                 _graph_out = self.graph_dense11(graph_text_out.transpose(1,2)).squeeze(-1)
#             elif _index == 1:
#                 _graph_out = self.graph_dense12(graph_text_out.transpose(1,2)).squeeze(-1)
#             elif _index == 2:
#                 _graph_out = self.graph_dense13(graph_text_out.transpose(1,2)).squeeze(-1)
            
#             if _index == 0:
#                 graph_out = _graph_out
#             else:
#                 graph_out = torch.concat((graph_out,_graph_out),-1)
        
#         # hidden_state,pooled_output= self.bert(fol_bert_indices, token_type_ids=fol_bert_type,attention_mask=fol_bert_mask,return_dict=False)
#         if self.with_text:
#             text_hidden_state,pooled_output= self.bert(ti, token_type_ids=tt,attention_mask=tm,return_dict=False)
#             text_hidden_state = text_hidden_state[mlm >= 0].view(text_hidden_state.size(0),1,text_hidden_state.size(-1)).squeeze(1)

#         # out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
#         # for index in range(hidden_state.size(0)):
#             # out[index] = evaluate_expression(fol_bert_indices[index],hidden_state[index]).squeeze()

        
#         if self.with_text:
#             out = torch.cat((graph_out,text_hidden_state),-1)
#             out = self.dropout(out)
#             out = self.dense1(out)
#         else:
#             out = self.dropout(graph_out)
#             out = self.dense2(out)

#         self.tokenizer.get_labels()
#         logits = [
#             torch.mm(
#                 out[:,:],
#                 self.bert.embeddings.word_embeddings.weight[i].transpose(1,0)
#             ) for i in self.tokenizer.prompt_label_idx
#         ]
#         # logits = self.dense(out)
#         # print(logits.size(), logits.device)
#         return logits