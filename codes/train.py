# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import logging
import argparse
import math
import os
import sys
import random
import numpy
import csv
from sklearn import metrics
from time import strftime, localtime

from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel,AdamW, \
                         AutoTokenizer, AutoModel, BartConfig, BartForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel
import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

# from data_utilsP import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
# from data_utilsP import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
# from models import LSTM, ASGCN,BERT_LSTM
from models.bert_spc import BERT_SPC
from models.lstm_fol import LSTM_FOL
from models.bart_fol import BART_FOL
from models.bert_fol_text import BERT_FOL_T
from models.bert_fol_text_graph import BERT_FOL_G
from models.bert_fol_text_rgraph import BERT_FOL_R
from models.roberta_fol_text import ROBERTA_FOL_T

transformers.logging.set_verbosity_error()

# BASEPATH = '/home/dingdaijun/data_list/dingdaijun/LTN_merge/datasets/kgpt/fol_bkl'
BASEPATH = '/home/dingdaijun/data_list/dingdaijun/LTN_merge/datasets/merge'

class Encoder(BartPretrainedModel):
    
    def __init__(self, config: BartConfig):
        
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder(config, self.shared)

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
    
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        LLM_name = BASEPATH.split('/')[-1]
        if '-' in self.opt.dataset and self.opt.dataset[-1] != '-':
            log_path = f'../log/{LLM_name}/{self.opt.model_name}_{self.opt.with_text}/cross/'
        elif self.opt.dataset[-1] == '-':
            log_path = f'../log/{LLM_name}/{self.opt.model_name}_{self.opt.with_text}/zero/'
        else:
            log_path = f'../log/{LLM_name}/{self.opt.model_name}_{self.opt.with_text}/in-domain/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger = get_logger(f'{log_path}{self.opt.dataset}_{self.opt.model_name}_{self.opt.lid}.log',name='normal')
        self.logger = logger
        best_logger = get_logger(f'{log_path}best_{self.opt.dataset}_{self.opt.model_name}_{self.opt.lid}.log',name='best')
        self.best_logger = best_logger

        print(opt)
        # assert False
        if 'roberta' in opt.model_name:
            roberta = RobertaModel.from_pretrained(opt.pretrained_model_name)
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_model_name,lid = opt.lid, model_name=opt.model_name,model=roberta)
            self.model = opt.model_class(roberta,opt,tokenizer).to(opt.device)
        elif 'bert' in opt.model_name:
            bert = BertModel.from_pretrained(opt.pretrained_model_name)
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_model_name,lid = opt.lid, model_name=opt.model_name,model=bert)
            self.model = opt.model_class(bert,opt,tokenizer).to(opt.device)
        elif 'bart' in opt.model_name:
            config = BartConfig.from_pretrained(opt.pretrained_model_name)
            bart = Encoder.from_pretrained(opt.pretrained_model_name)
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_model_name,lid = opt.lid, model_name=opt.model_name,model=bart)
            self.model = opt.model_class(opt,config,bart,tokenizer).to(opt.device)

        # elif 'twitter' in opt.pretrained_model_name and 'bert' in opt.model_name:
        #     bert = AutoModel.from_pretrained(opt.pretrained_model_name) 
        #     tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_model_name) #!!


        #     self.model = opt.model_class(bert, opt).to(opt.device)
        #     # self.opt.num_epoch = 15
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='../dat/{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='../dat/{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer,opt,'train')
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer,opt,'test')

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        if opt.use_prompt == True:
            self.A_top = min(5,len(tokenizer.get_label_words(self.opt.lid)[0]))
            self.F_top = min(5,len(tokenizer.get_label_words(self.opt.lid)[1]))
            self.N_top = min(5,len(tokenizer.get_label_words(self.opt.lid)[2]))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in vars(self.opt):
            self.logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for n, p in self.model.named_parameters():
            if 'bart' not in n and 'bert' not in n:
                if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
            # elif type(child) == BertModel:
            #     print('='*20)
            #     print('freeze bert now')
            #     unfreeze_layers = ['layer.10','layer.11','bert.pooler','out.']
            #     # for name, param in child.named_parameters():

            
            #     for name ,param in child.named_parameters():
            #         param.requires_grad = False
            #         for ele in unfreeze_layers:
            #             if ele in name:
            #                 param.requires_grad = True
            #                 break


    def _train(self, criterion, optimizer, train_data_loader, val_data_loader,test_data_loader):
        best_val = []
        best_test = []
        max_maf1a = 0
        max_val_epoch = 0
        # global_step = 0
        path = None
        if self.opt.log_step == -1:
            self.opt.log_step = len(train_data_loader) // 5
        for i_epoch in range(self.opt.num_epoch):
            self.logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            t_batch = 0
            for i_batch, batch in enumerate(train_data_loader):
                self.model.train()
                t_batch += 1
                # global_step += 1
                optimizer.zero_grad()
                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = batch['polarity'].to(self.opt.device)

                outputs = self.model(inputs)
                if self.opt.use_prompt == True:
                    res = torch.zeros(outputs[0].size(0), int(self.opt.polarities_dim))
                    res = res.cuda()
                    res[:,0] = torch.sum(outputs[0].topk(self.A_top, dim=1, largest=True).values, axis=-1) / self.A_top
                    res[:,1] = torch.sum(outputs[1].topk(self.F_top, dim=1, largest=True).values, axis=-1) / self.F_top
                    if int(self.opt.polarities_dim) == 3:
                        res[:,2] = torch.sum(outputs[2].topk(self.N_top, dim=1, largest=True).values, axis=-1) / self.N_top
                    # loss += criterion(res, labels)
                    outputs = res
                loss = criterion(outputs, targets)
                # print(i_batch)
                
                if torch.isnan(loss):
                    print(i_batch,batch['bert_fol_inputs'])
                    print('===============================')
                    # continue
                    assert False
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if t_batch % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    self.logger.info(f'{i_batch}/{len(train_data_loader)}\ttrain_loss: {train_loss}\ttrain_acc: {round(train_acc * 100,2)}')

                    val_acc, val_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a = self._evaluate_acc_f1(val_data_loader)
                    self.logger.info(f'Val: ma_f1: {round(val_f1*100,2)}\tacc: {round(val_acc*100,2)}\tavg_f1: {round(avg_f1*100,2)}\tma_all_f1: {round(maf1a*100,2)}\tmi_all_f1: {round(mif1a*100,2)}\tfavor_f1: {round(f_f1*100,2)}\tagainst_f1: {round(a_f1*100,2)}\tnone_f1: {round(n_f1*100,2)}')
                    if val_f1 > max_maf1a:
                        best_val = [val_acc,val_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a,i_epoch]
                        max_maf1a = val_f1
                        max_val_epoch = i_epoch
                        test_acc, test_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a = self._evaluate_acc_f1(test_data_loader)
                        best_test = [test_acc, test_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a,i_epoch]
                        self.logger.info(f'Test: ma_f1: {round(test_f1*100,2)}\tacc: {round(test_acc*100,2)}\tavg_f1: {round(avg_f1*100,2)}\tma_all_f1: {round(maf1a*100,2)}\tmi_all_f1: {round(mif1a*100,2)}\tfavor_f1: {round(f_f1*100,2)}\tagainst_f1: {round(a_f1*100,2)}\tnone_f1: {round(n_f1*100,2)}')
                        if self.opt.save_model:
                            path = f'../state_dict/{self.opt.model_name}_{self.opt.dataset}'
                            if not os.path.exists('../state_dict'):
                                os.makedirs('../state_dict')
                            torch.save(self.model.state_dict(), path)
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
        self.logger.info(f'Best_test: epoch:{best_test[-1]}\tma_f1: {round(best_test[1]*100,2)}\tacc: {round(best_test[0]*100,2)}\tavg_f1: {round(best_test[2]*100,2)}\tma_all_f1: {round(best_test[6]*100,2)}\tmi_all_f1: {round(best_test[7]*100,2)}\tfavor_f1: {round(best_test[3]*100,2)}\tagainst_f1: {round(best_test[4]*100,2)}\tnone_f1: {round(best_test[5]*100,2)}')
        self.best_logger.info(f'Best_test: epoch:{best_test[-1]}\tma_f1: {round(best_test[1]*100,2)}\tacc: {round(best_test[0]*100,2)}\tavg_f1: {round(best_test[2]*100,2)}\tma_all_f1: {round(best_test[6]*100,2)}\tmi_all_f1: {round(best_test[7]*100,2)}\tfavor_f1: {round(best_test[3]*100,2)}\tagainst_f1: {round(best_test[4]*100,2)}\tnone_f1: {round(best_test[5]*100,2)}')
        self.logger.info(f'Best_val: epoch:{best_val[-1]}\tma_f1: {round(best_val[1]*100,2)}\tacc: {round(best_val[0]*100,2)}\tavg_f1: {round(best_val[2]*100,2)}\tma_all_f1: {round(best_val[6]*100,2)}\tmi_all_f1: {round(best_val[7]*100,2)}\tfavor_f1: {round(best_val[3]*100,2)}\tagainst_f1: {round(best_val[4]*100,2)}\tnone_f1: {round(best_val[5]*100,2)}')
        
        return path

    def _evaluate_acc_f1(self, data_loader):
        t_targets_all, t_outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]

                t_outputs = self.model(t_inputs)
                if self.opt.use_prompt == True:
                    res = torch.zeros(t_outputs[0].size(0), int(self.opt.polarities_dim))
                    res = res.cuda()
                    res[:,0] = torch.sum(t_outputs[0].topk(self.A_top, dim=1, largest=True).values, axis=-1) / self.A_top
                    res[:,1] = torch.sum(t_outputs[1].topk(self.F_top, dim=1, largest=True).values, axis=-1) / self.F_top
                    if int(self.opt.polarities_dim) == 3:
                        res[:,2] = torch.sum(t_outputs[2].topk(self.N_top, dim=1, largest=True).values, axis=-1) / self.N_top
                    t_outputs = res
                t_targets = t_batch['polarity'].to(self.opt.device)
                if t_targets_all is None:
                    t_targets_all = t_targets.cpu()
                    t_outputs_all = t_outputs.cpu()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.cpu()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.cpu()), dim=0)
        acc = metrics.accuracy_score(t_targets_all, torch.argmax(t_outputs_all, -1))
        maf1a = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro')
        mif1a = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='micro')
        maf1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro',labels=[0,1])
        mif1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='micro',labels=[0,1])
        f_f1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro',labels=[1])
        a_f1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro',labels=[0])
        n_f1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro',labels=[2])
        avg_f1 = (mif1 + maf1)/2
        return acc, maf1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a

    def run(self):

        criterion = nn.CrossEntropyLoss()

        _params = [
            # {'params': [p for p in child.parameters()]} for child in self.model.children() if type(child) != BertModel and type(child) != RobertaModel and type(child) != AutoModel
            {'params': [p for n,p in child.named_parameters()],'lr':self.opt.lr} for child in self.model.children() if type(child) != BertModel and type(child) != RobertaModel and type(child) != AutoModel
        ]

        no_decay = ['bias', 'LayerNorm.weight']
        if 'bert' in self.opt.model_name:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5,'lr':self.opt.bert_lr,'eps':1e-8},
                {'params': [p for n, p in self.model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':self.opt.bert_lr,'eps':1e-8},
            ]
            optimizer_grouped_parameters += _params
        elif 'bart' in self.opt.model_name:
            for n, p in self.model.named_parameters():
                if "bart.shared.weight" in n or "bart.encoder.embed" in n:
                    p.requires_grad = False
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if n.startswith('bart.encoder.layer')] , 'lr': self.opt.bert_lr},
                {'params': [p for n, p in self.model.named_parameters() if not n.startswith('bart.encoder.layer')] , 'lr': self.opt.lr},
                ]
        else:
            optimizer_grouped_parameters = _params

        optimizer = self.opt.optimizer(optimizer_grouped_parameters)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader,test_data_loader)
        # best_model_path = f'../state_dict/{self.opt.model_name}_{self.opt.dataset}'
        if self.opt.save_model:
            self.model.load_state_dict(torch.load(best_model_path))
            test_acc, test_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a = self._evaluate_acc_f1(test_data_loader)
            self.logger.info(f'Best_test: ma_f1: {round(test_f1*100,2)}\tacc: {round(test_acc*100,2)}\tavg_f1: {round(avg_f1*100,2)}\tma_all_f1: {round(maf1a*100,2)}\tmi_all_f1: {round(mif1a*100,2)}\tfavor_f1: {round(f_f1*100,2)}\tagainst_f1: {round(a_f1*100,2)}\tnone_f1: {round(n_f1*100,2)}')



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_fol', type=str)
    parser.add_argument('--dataset', default='dt', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--bert_lr', default=1e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=-1, type=int)
    parser.add_argument('--pretrained_model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=40, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)                  
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--save_model', default=False, type=bool)
    parser.add_argument('--seed', default=2023, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--pid', default=1, type=int)
    parser.add_argument('--tid', default=1, type=int)
    parser.add_argument('--lid', default=0, type=int)
    parser.add_argument('--llambda', default=0.5, type=float)
    parser.add_argument('--use_prompt', default=True, type=bool)
    parser.add_argument('--with_text', default=False, type=bool)
    parser.add_argument('--nodes_num', default=20, type=int)
    parser.add_argument('--edge_num', default=40, type=int)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        # 'lstm': LSTM,
        # 'bert_lstm': BERT_LSTM,
        'bert': BERT_SPC,
        # 'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'bart_fol': BART_FOL,
        'lstm_fol': LSTM_FOL,
        'bert_fol_text': BERT_FOL_T,
        'bert_fol_graph': BERT_FOL_G,
        'bert_fol_rgraph': BERT_FOL_R,
        'roberta_fol_text': ROBERTA_FOL_T,
    }
    def dataset_files(data_target):
        if '-' not in data_target:
            
            return {
                'train': f'{BASEPATH}/{data_target}/train.csv',
                'test': f'{BASEPATH}/{data_target}/test.csv',
            }
        else:
            source_target, destin_target = data_target.split('-')
            if source_target != '' and destin_target != '':
                return {
                    'train': f'{BASEPATH}/{source_target}/{source_target}.csv',
                    'test': f'{BASEPATH}/{destin_target}/{destin_target}.csv',
                }
            else:
                all_target = ['dt','la','fm','hc']
                destin_target = next(filter(lambda x: x != '', [source_target,destin_target]))
                all_target.remove(destin_target)
                assert len(all_target) == 3
                return {
                    'train': [
                        f'{BASEPATH}/{all_target[0]}/{all_target[0]}.csv',
                        f'{BASEPATH}/{all_target[1]}/{all_target[1]}.csv',
                        f'{BASEPATH}/{all_target[2]}/{all_target[2]}.csv',
                        ],
                    'test': f'{BASEPATH}/{destin_target}/{destin_target}.csv',
                }


    input_colses = {
        'bert_fol_text': ['bert_fol_inputs','bert_fol_type','bert_fol_mask','bert_text_inputs','bert_text_type','bert_text_mask','mlm_labels'],
        'bert_fol_graph': ['bert_fol_inputs','bert_fol_type','bert_fol_mask','bert_text_inputs','bert_text_type','bert_text_mask','mlm_labels','graph_matrix','graph_text'],
        'bert_fol_rgraph': ['bert_fol_inputs','bert_fol_type','bert_fol_mask','bert_text_inputs','bert_text_type','bert_text_mask','mlm_labels','graph_index','graph_type','graph_text'],
        
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'bert_fol': ['bert_fol_inputs','bert_fol_type','bert_fol_mask'],
        'bart_fol': ['bert_fol_inputs','bert_fol_mask'],
        'bert_fol_respect': ['bert_fol_inputs','bert_fol_type','bert_fol_mask','bert_fol_pred_inputs','bert_fol_pred_type','bert_fol_pred_mask'],
        # 'bert_fol_text': ['bert_fol_inputs','bert_fol_mask','bert_fol_pred_inputs','bert_fol_pred_mask','bert_text_inputs','bert_text_mask','mlm_labels'],
        'roberta_fol_text': ['bert_fol_inputs','bert_fol_mask','bert_fol_pred_inputs','bert_fol_pred_mask','bert_text_inputs','bert_text_mask','mlm_labels'],
        'roberta_fol_respect': ['bert_fol_inputs','bert_fol_mask','bert_fol_pred_inputs','bert_fol_pred_mask'],
        'lstm_fol': ['fol_indices'],
        # 'aen_bert': ['fol_indices'],

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw':AdamW,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files(opt.dataset)
    opt.inputs_cols = input_colses[f'{opt.model_name}']
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
