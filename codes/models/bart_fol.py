import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, BartForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel
from layers.operate_bart import evaluate_expression
'''
∀ 47444, 7471
 x 3023
 ¬ 5505, 11582 | 11582
 ∨ 47444, 11423  | 50266
 → 42484  |50267
 ) 4839
 ( 36
( 1640
 ∧ 47444, 6248  | 50265
∧ ∨ ¬ →
 '''


class BART_FOL(nn.Module):

    def __init__(self,opt,config,bart,tokenizer):

        super(BART_FOL, self).__init__()
        num_labels, dropout = opt.polarities_dim, opt.dropout
        self.dropout = nn.Dropout(dropout) 
        self.relu = nn.GELU()
        
        self.config = config
        self.bart = bart
        self.bart.pooler = None

        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.out1 = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size)
        
        self.notnetw = nn.Sequential(
            nn.LeakyReLU(negative_slope=-0.2),
            nn.Linear(self.bart.config.hidden_size,self.bart.config.hidden_size),
                                    )
        self.andnetw = nn.Sequential(
            nn.Linear(self.bart.config.hidden_size,1),
                                    )
        self.ornetw = nn.Sequential(
            nn.Linear(self.bart.config.hidden_size*2,self.bart.config.hidden_size),
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
    def forward(self, input):
        
        x_input_ids, x_atten_masks,text_input,text_mask = input
        hidden_state = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)[0]
        text_hidden_state = self.bart(input_ids=text_input, attention_mask=text_mask)[0]
        # print(hidden_state.size())
        # assert False
        # out = self.linear(last_hidden.transpose(1,2)).squeeze(2)
        # # out = nn.GELU()(self.fc(out))
        # # # out = last_hidden[0]
        out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
        for index in range(hidden_state.size(0)):
            out[index] = evaluate_expression(x_input_ids[index],hidden_state[index],self.andnet,self.ornet,self.notnet).squeeze()
        # print(torch.mean(text_hidden_state,-2).size())
        out = torch.cat((out,torch.mean(text_hidden_state,-2)),-1)
        out = self.dropout(out)
        out = self.out1(out)
        out = self.out(out)
        # print(out.size())
        # print(torch.mean(hidden_state,dim=1).size())
        # out = self.out(torch.mean(hidden_state,dim=1))
        # print(out.size())
        
        
        return out
        #         self.tokenizer.get_labels()
        # # for i in self.tokenizer.prompt_label_idx:
        # #     print(i)
        # #     print('2xx',self.bart.encoder.embed_tokens.weight[i].transpose(1,0),self.bart.encoder.embed_tokens.weight[i].transpose(1,0).size())
        # #     print('1xx',self.bart(torch.tensor([i]).cuda())[0].detach().transpose(1,0),self.bart(torch.tensor([i]).cuda())[0].detach().transpose(1,0).size())
        # # assert False
        # logits = [
        #     torch.mm(
        #         out[:,:],
        #         # self.bert.embeddings.word_embeddings.weight[i].transpose(1,0)
        #         # self.bart(torch.tensor([i]).cuda())[0].squeeze(0).transpose(1,0)
        #         i.transpose(1,0).cuda()
        #         # self.bart.encoder.embed_tokens.weight[i].transpose(1,0)
        #     ) for i in self.tokenizer.prompt_label_embed
        # ]
        # # print(torch.mean(hidden_state,dim=1).size())
        # # out = self.out(torch.mean(hidden_state,dim=1))
        # # print(out.size())
        
        
        # return logits