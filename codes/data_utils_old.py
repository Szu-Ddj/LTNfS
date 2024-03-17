# 舍去v关系，根据连接关系构成一张图并返回邻接矩阵
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, \
                         AutoTokenizer
import csv
import random
import networkx as nx
# import matplotlib.pyplot as plt
import ast
from senticnet.senticnet import SenticNet
SN = SenticNet()
pattern = r'(\w+|\S)'
def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        dataname = ''
        for fname in fnames:
            if type(fname) == list:
                for sign_fname in fname:
                    print(sign_fname)
                    print(sign_fname.split('/')[-1].split('_')[0])
                    dataname += sign_fname.split('/')[-1].split('_')[0]
                    with open(sign_fname,'r') as f:
                        l1s = csv.DictReader(f)
                        for l1 in l1s:

                            fol = re.sub(r'∃\w|∀\w','',f"({l1['FOL']} → {l1['Pred']})").strip()
                            fol = re.sub(r'\((\w)\)', r' \1',fol)
                            fol = re.sub(r'\s+',' ',fol)
                            fol = ' '.join(re.findall(pattern, fol))

                            text += l1['Tweet'] + ' ' + fol + ' ' + l1['Target'] + ' ' + l1['FOLC']
            else:
                print(fname)
                print(fname.split('/')[-2])
                dataname += fname.split('/')[-2]
                with open(fname,'r') as f:
                    l1s = csv.DictReader(f)
                    for l1 in l1s:
                        fol = re.sub(r'∃\w|∀\w','',f"({l1['FOL']} → {l1['Pred']})").strip()
                        fol = re.sub(r'\((\w)\)', r' \1',fol)
                        fol = re.sub(r'\s+',' ',fol)
                        fol = ' '.join(re.findall(pattern, fol))
                        text += l1['Tweet'] + ' ' + fol + ' ' + l1['Target'] + ' ' + l1['FOLC']

        tokenizer = Tokenizer(max_seq_len,dataname)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
        # print('Write the Special Tokens and the Restart the Program')
        # assert False
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else '/home/dingdaijun/data_list/dingdaijun/glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


'''
# ∀ 1
# ∃ 2
( 1
) 2
¬ 3
∨ 4
∧ 5
→ 6
'''
class Tokenizer(object):
    def __init__(self, max_seq_len, dataname = 'template',lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {'(':1,')':2,'¬':3,'∨':4,'∧':5,'→':6}
        self.idx2word = {1:'(',2:')',3:'¬',4:'∨',5:'∧',6:'→',}
        self.idx = 7
        self.dataname = dataname
    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        words_num = {}
        for word in words:
            if words_num.get(word) == None:
                words_num[word] = 0
            words_num[word] += 1
        for word in words_num:
            if word not in self.word2idx and words_num[word]:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
        # print(self.word2idx)
        with open(f'../dat/{self.dataname}_special_idx.txt','w') as fw:
            fw.write(f"(: {self.word2idx['(']}\n")
            fw.write(f"): {self.word2idx[')']}\n")
            # fw.write(f"∀: {self.word2idx['∀']}\n")
            # fw.write(f"∃: {self.word2idx['∃']}\n")
            fw.write(f"¬: {self.word2idx['¬']}\n")
            fw.write(f"∨: {self.word2idx['∨']}\n")
            fw.write(f"∧: {self.word2idx['∧']}\n")
            fw.write(f"→: {self.word2idx['→']}\n")
        print('tokenizer is over:', len(self.word2idx))
        

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len,pretrained_model_name,lid=0, tid=1,model_name='NPS',model=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name) #!!

        self.max_seq_len = max_seq_len
        self.tokenizer.add_tokens([])
        self.lid = lid
        self.model = model
        self.temps = self.get_prompt_template()

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', add = 0,mmodel=''):

        if mmodel == 'fol':
            truc = True
        elif mmodel == 'prompt':
            truc = False
        else:
            print('='*30)
            print(mmodel,text)
            assert False
            
        if len(text) == 1:
            # sequence = self.tokenizer.encode_plus(text[0],max_length=self.max_seq_len+add,padding='max_length',truncation=truc,return_tensors='pt',add_special_tokens=False).values()
            sequence = self.tokenizer.encode_plus(text[0],max_length=self.max_seq_len+add,padding='max_length',truncation=truc,return_tensors='pt').values()
        else:
            sequence = self.tokenizer.encode_plus(text[0],text_pair=text[1],max_length=self.max_seq_len+add,padding='max_length',truncation=truc,return_tensors='pt').values()
        sequence = [item.squeeze(0) for item in sequence]
        if len(sequence[0]) > self.max_seq_len + add:
            # print("===========it's Prompt==========")
            for index in range(len(sequence)):
                sequence[index] = sequence[index][len(sequence[index]) - self.max_seq_len - add:]
        return sequence
        # sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        # if len(sequence) == 0:
        #     sequence = [0]
        # if reverse:
        #     sequence = sequence[::-1]
        # if len(sequence) - add > self.max_seq_len:
        #     print('***length out***', len(sequence) - add - self.max_seq_len, 'add: ',add)
        # return pad_and_truncate(sequence, self.max_seq_len + add, padding=padding, truncating=truncating)
    def get_label_words(self,lid = 0):
        if lid == 0:
            return [
                ['opposed'],
                ['support'],
                ['neutral']
            ]
        elif lid == 1:
            all_seed_words = [
                ['opposed'],
                ['support'],
                ['neutral']
            ]
            return self.get_senticnet_n_hop(all_seed_words, 1, self.tokenizer)
        elif lid == 2:
            all_seed_words = [
                ['opposed'],
                ['support'],
                ['neutral']
            ]
            return self.get_senticnet_n_hop(all_seed_words, 2, self.tokenizer)
        elif lid == 3:
            all_seed_words = [
                ['opposed'],
                ['support'],
                ['neutral']
            ]
            return self.get_senticnet_n_hop(all_seed_words, 3, self.tokenizer)
        elif lid == 4:
            return [
                    ["wrong", "bad", "stupid"],
                    ["beautiful", "good", "great"],
                    ["neutral", "unique", "cool"],
                ]

        else:
            raise Exception("label_id error, please choose correct id")
        
    def get_labels(self):
        self.verbalizer = self.get_label_words(self.lid)
        mask_ids = []
        mask_pos = 0
        self.temps = 'Is a template {mask}'
        temp = self.temps.format(text_a = 'a', text_b = 'a',classi = 'a',reason = 'a', mask = self.tokenizer.mask_token)
        temp = temp.split(' ')
        # if '' in temp:
        #     temp.remove('')
        _temp = temp.copy()
        
        original = self.tokenizer.encode(' '.join(temp), add_special_tokens = False)
        
        for i in range(len(_temp)):
            if _temp[i] == self.tokenizer.mask_token:
                _mask_pos = i
        for i in range(len(original)):
            if original[i] == self.tokenizer.mask_token_id:
                mask_pos = i

        sign = True
        for index, name in enumerate(self.verbalizer):
            mask_id = []
            for item in name:
                _temp[_mask_pos] = item
                final = self.tokenizer.encode(' '.join(_temp), add_special_tokens = False)
                if len(final) != len(original):
                    sign = False
                mask_id.append(final[mask_pos])
            mask_ids.append(mask_id)
        assert sign
        self.prompt_label_idx = mask_ids
        return self.prompt_label_idx
    def get_prompt_template(self,tid = 2):
        if tid == 1:
            temp = "{text_a} . From the perspective of {classi}, {reason} The attitude to {text_b} is {mask} ."
        elif tid == 2:
            temp = '{text_a} The attitude to {text_b} is {mask} .'
        elif tid == 3:
            temp = '{text_a} is {mask} of {text_b} .'
        elif tid == 4:
            temp = '{text_a} . The {text_b} made me feel {mask} .'
        elif tid == 5:
            temp = '{text_a} . Is it irony ? {mask} .'
        else:
            raise Exception("template_id error, please choose correct id")
        return temp
    

class build_FOL_graph:
    def __init__(self):
        self.pattern = r'(\w+|\S)'
        self.map_weight = {True:-1,False:1}
        self.stopwords_list = []
        with open('../datasets/stopwords.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.stopwords_list.append(line.strip())

    def build_graph(self,graph,prev,value,opera_sign):
        graph.add_nodes_from([item.lower().replace('¬','') for item in prev if len(item) >= 1])
        graph.add_nodes_from([item.lower().replace('¬','') for item in value if len(item) >= 1])
        if opera_sign == '∧':
            graph.add_edges_from([(itemx.lower().replace('¬',''),itemy.lower().replace('¬',''),{'weight':itemy.count('¬')%2}) for itemx in prev for itemy in value if itemx != itemy and len(itemx)>=1 and len(itemy)>=1])
            graph.add_edges_from([(itemy.lower().replace('¬',''),itemx.lower().replace('¬',''),{'weight':itemx.count('¬')%2}) for itemx in prev for itemy in value if itemx != itemy and len(itemx)>=1 and len(itemy)>=1])
        elif opera_sign == '→':
            graph.add_edges_from([(itemx.lower().replace('¬',''),itemy.lower().replace('¬',''),{'weight':itemy.count('¬')%2}) for itemx in prev for itemy in value if itemx != itemy and len(itemx)>=1 and len(itemy)>=1])
    
    def fit_text(self,original_string):
        match = re.search(r'¬\((.*?)\)', original_string)

        if match:
            # 获取括号内的内容
            inside_content = f"¬{match.group(1)}"
            new_content = re.sub(r'(\∨|\→|∧)', r'\1 ¬', inside_content)

            # 替换原始字符串
            result_string = original_string.replace(match.group(0), f'({new_content})')

            return result_string
        else:
            return original_string
    def clean_data(self,text):
        text = re.sub(r'\((\w)\)',r' ',text)
        text = text.replace('¬∀x','∀x ¬')
        text = text.replace('¬∃x','∀x ¬')
        text = text.replace('¬∃y','∀y ¬')
        text = text.replace('¬∃y','∀y ¬')
        text = text.replace('¬','¬ ')
        text = text.replace('(','( ')
        text = re.sub(r'∃\w|∀\w','',text).strip()
        text_list = text.split(' ')
        text = ' '.join([item for item in text_list if item not in self.stopwords_list])
        # return text
        text = re.sub(r'\s+',' ',' '.join(re.findall(self.pattern, text))).replace('¬ ','¬')
        cnt = 0
        # print(text)
        while '¬(' in text and cnt <=3:
            # print(text)
            text = self.fit_text(text)
            cnt += 1

        return re.sub(r'\s+',' ',' '.join(re.findall(self.pattern, text))).replace('¬ ','¬')
            
        # text = re.split(r'(\∧|\→|\∨)', text)
        # return [item.strip() for item in text if item.strip()]

        # text = re.sub('')
    def split_opera_sign(self,text):
        text = re.split(r'(\∧|\→|\∨)', text)
        return [item.strip() for item in text if item.strip()]

    def graph2matrix(self,graph):
        ad_matrix = nx.to_numpy_matrix(graph)
        word_nodes = graph.nodes
        try:
            assert ad_matrix.shape[0] == len(word_nodes)
        except:
            print(ad_matrix)
            print(word_nodes)
            assert False
        return np.asarray(ad_matrix),list(word_nodes)

    
    def run_build(self,line):
        _all_node = []
        _line = line
        line = self.clean_data(line.replace('>','∧'))
        line = line.replace('¬ ','¬')
        G = nx.DiGraph()
        str_stack = []
        line = f'( {line} )'
        # print(line)
        for _index,item in enumerate(line.split(' ')):
            # print(item,str_stack)
            # print(_index == len(line.split(' '))-1)
            if item != ')':
                str_stack.append(item)
            else:
                # print(item,str_stack)
                sub_fol = []
                # print(str_stack)
                while str_stack[-1] != '(':
                    sub_fol.append(str_stack[-1])
                    str_stack.pop()
                    if len(str_stack) == 0: break
                try:
                    str_stack.pop()
                except:
                    # print(_line)
                    pass
                try:
                    assert sub_fol[0] not in ['∧','→','∨']
                except:
                    print(_line)
                    print(sub_fol)
                    print(line)
                    assert False
                sub_fol.reverse()
                sub_fol = self.split_opera_sign(' '.join([str(item) for item in sub_fol]))
                # print(sub_fol)
                try:
                    sub_fol = [ast.literal_eval(item) if item.startswith('[') else item for item in sub_fol]
                except:
                    print(_line)
                    print(sub_fol)
                    assert False
                prev = []
                value = []
                pre_sign = ''
                for _item in sub_fol:
                    # if len(_item) < 1:
                    #     continue
                    if _item not in ['∧','→','∨']:
                        if type(_item) == list:
                            value = _item
                        else:
                            value= [_item]
                        _all_node.extend(value)
                    else:
                        if len(prev) != 0:
                            self.build_graph(G,prev,value,pre_sign)
                        # else:
                        pre_sign = _item
                        prev = value
                        value = []
                if len(prev) != 0 and len(value) != 0:
                    self.build_graph(G,prev,value,pre_sign)
                str_stack.append(_all_node)
                _all_node = []
        return self.graph2matrix(G)

import csv
import random
import re
class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer,opt,tname):
        all_data = []
        pattern = r'(\w+|\S)'
        BG = build_FOL_graph()
        if 'vast' in opt.dataset:
            match = {'AGAINST':0,'FAVOR':1,'NONE':2,'0':0,'1':1,'2':2,'support':1,'opposed':0,'neutral':2}
        else:
            match = {'AGAINST':0,'FAVOR':1,'NONE':2,'-1':0,'1':1,'0':2,'support':1,'opposed':0,'neutral':2}
        
        def graph_preModel(graph_matrix_text_list,max_length = 20):
            graph_matrix,text_list = graph_matrix_text_list
            np.fill_diagonal(graph_matrix, 1)
            _graph_matrix = graph_matrix.copy()
            PM_tokenizer = tokenizer.tokenizer
            row_index = 0
            fol_graph_indices = []
            try:
                assert graph_matrix.shape[0] == len(text_list)
            except:
                print(graph_matrix, text_list)
                assert False
            for _,text in enumerate(text_list):
                fol_graph_indices.extend(PM_tokenizer.encode(text,add_special_tokens=False))
                len_PMTtext = len(PM_tokenizer.encode(text,add_special_tokens=False))
                while len_PMTtext > 1:
                    new_row = graph_matrix[row_index, :]
                    graph_matrix = np.insert(graph_matrix, row_index, new_row, axis=0)

                    new_col = graph_matrix[:, row_index]
                    graph_matrix = np.insert(graph_matrix, row_index, new_col, axis=1)
                    len_PMTtext -= 1
                    row_index += 1
                row_index += 1
            if graph_matrix.shape[0] != len(fol_graph_indices):
                print(graph_matrix,fol_graph_indices)
                print(_graph_matrix,text_list)
                assert False
            if len(fol_graph_indices) >= max_length:
                graph_matrix = graph_matrix[:max_length,:max_length]
                fol_graph_indices = torch.tensor(fol_graph_indices[:max_length])
            else:
                graph_matrix = np.vstack([graph_matrix, np.zeros((max_length-len(fol_graph_indices), graph_matrix.shape[1]))])
                graph_matrix = np.hstack([graph_matrix, np.zeros((graph_matrix.shape[0], max_length-len(fol_graph_indices)))])
                fol_graph_indices = torch.tensor(fol_graph_indices + [0]*(max_length-len(fol_graph_indices)))
            # if graph_matrix.shape[0] != 20:
            #     print(graph_matrix.shape,fol_graph_indices.size())
            return graph_matrix,fol_graph_indices

            


        def deal_data(sign_fname):
            sall_data = []
            cnt = 1
            with open(sign_fname,'r',encoding='utf-8') as f:
                lines = csv.DictReader(f)
                for index,line in enumerate(lines):
                    text = line['Tweet']
                    # line['Attitude'] = line['Stance']
                    # line['Reason'] = 'none'

                    # fol = line['FOL']

                    # fol = re.sub(r'∃\w|∀\w','',f"({line['FOL']})").strip()
                    if len(line['FOL']) <= 2:
                        fol = 'empty'
                    else:
                        # fol = re.sub(r'∃\w|∀\w','',f"({line['FOL']} → {line['Pred']})").strip()
                        fol = re.sub(r'∃\w|∀\w','',f"({line['FOL']})").strip()

                        # fol = re.sub(r'\((\w)\)', r' \1',fol)
                        fol = re.sub(r'\((\w)\)', r' ',fol)
                        fol = re.sub(r'\s+',' ',fol)
                        fol = ' '.join(re.findall(pattern, fol))
                    # fol_pred = line['Pred']
                    # fol = line['FOL'] + line['FOLC']
                    target = line['Target']
                    
                    ad_matrix_gpt,word_nodes_gpt = graph_preModel(BG.run_build(line['Ans1']))
                    ad_matrix_claude,word_nodes_claude = graph_preModel(BG.run_build(line['Ans2']))
                    ad_matrix_lla,word_nodes_lla = graph_preModel(BG.run_build(line['Ans3']))

                    polarity = match[line['Stance'].upper()]
                    if 'bart' in opt.model_name or 'bert' in opt.model_name:
                        prompt_template = tokenizer.temps.format(text_a = text, text_b = target,mask = tokenizer.tokenizer.mask_token)

                        bert_text = tokenizer.text_to_sequence([prompt_template],add=120,mmodel='prompt')
                        mlm_labels = np.where(bert_text[0] == tokenizer.tokenizer.mask_token_id,1,-1)
                        # bert_text_fol = tokenizer.text_to_sequence([text, fol])
                        # bert_fol_target = tokenizer.text_to_sequence([fol,target])
                        # bert_text_fol_target = tokenizer.text_to_sequence([text + fol,target])
                        bert_fol = tokenizer.text_to_sequence([fol,target],add=80,mmodel='fol')
                        cnt += 1

                        # print(ad_matrix_gpt.shape,ad_matrix_claude.shape, ad_matrix_lla.shape)
                        assert ad_matrix_gpt.shape[0] == 20 == ad_matrix_claude.shape[0] == ad_matrix_lla.shape[0]
                        # print(torch.tensor([ad_matrix_gpt.tolist(),ad_matrix_claude.tolist(),ad_matrix_lla.tolist()]).long().size())
                        if 'bert' in opt.pretrained_model_name and 'roberta' not in opt.pretrained_model_name:
                            data = {
                                'bert_text_inputs': bert_text[0],
                                'bert_text_type': bert_text[1],
                                'bert_text_mask': bert_text[2],

                                # 'bert_text_fol_inputs': bert_text_fol[0],
                                # 'bert_text_fol_type': bert_text_fol[1],
                                # 'bert_text_fol_mask': bert_text_fol[2],
                                'bert_fol_inputs': bert_fol[0],                            
                                'bert_fol_type': bert_fol[1],                            
                                'bert_fol_mask': bert_fol[2], 

                                'graph_matrix': torch.tensor([ad_matrix_gpt.tolist(),ad_matrix_claude.tolist(),ad_matrix_lla.tolist()]).float(),
                                # 'graph_matrix': torch.tensor([torch.tensor(ad_matrix_gpt).long(),torch.tensor(ad_matrix_claude).long(),torch.tensor(ad_matrix_lla).long()]).long(),
                                'graph_text': torch.tensor([word_nodes_gpt.tolist(),word_nodes_claude.tolist(),word_nodes_lla.tolist()]).long(), 

                                'mlm_labels': mlm_labels, 

                                'polarity': polarity,
                            }
                        else:
                            data = {
                                'bert_text_inputs': bert_text[0],
                                'bert_text_mask': bert_text[1],

                                # 'bert_text_fol_inputs': bert_text_fol[0],
                                # 'bert_text_fol_mask': bert_text_fol[1],

                                'bert_fol_inputs': bert_fol[0],                            
                                'bert_fol_mask': bert_fol[1],    
 
                                'mlm_labels': mlm_labels,
                                'polarity': polarity,
                            }
                    else:
                        text_target_indices = tokenizer.text_to_sequence(text + ' ' + target)
                        text_fol_indices = tokenizer.text_to_sequence(text + ' ' + fol)
                        text_indices = tokenizer.text_to_sequence(text)
                        fol_indices = tokenizer.text_to_sequence(fol)
                        fol_target_indices = tokenizer.text_to_sequence(fol + ' ' + target)
                        text_fol_target_indices = tokenizer.text_to_sequence(text + ' ' + fol + ' ' + target)
                        data = {
                            'text_target_indices': text_target_indices,
                            'text_fol_indices': text_fol_indices,
                            'text_fol_target_indices': text_fol_target_indices,
                            'text_indices': text_indices,
                            'fol_indices': fol_indices,
                            'fol_target_indices': fol_target_indices,

                            'polarity': polarity,
                        }
                    # print(data)
                    # assert False
                    sall_data.append(data)
            return sall_data


        if type(fname) == list:
            for sign_fname in fname:
                sall_data = deal_data(sign_fname)
                all_data.extend(sall_data)
        else:
            sall_data = deal_data(fname)
            all_data.extend(sall_data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

