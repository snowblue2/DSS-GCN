import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from layers.dynamic_rnn import DynamicLSTM

""""
embedding 去掉位置权重，改为softmax
"""

class GraphConvolution(nn.Module):
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
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ablation(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ablation, self).__init__()
        self.opt = opt
        self.usebert = self.opt.usebert
        if self.usebert:
            self.embed = BertModel.from_pretrained('./datasets/bert-base-uncased').requires_grad_(False)
            self.embed.eval()
            self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
            self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc5 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc6 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc7 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc8 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)


        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.dfc = nn.Linear(4*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(opt.dropout)
        self.weight = nn.Parameter(torch.FloatTensor(4 * opt.hidden_dim, 4 * opt.hidden_dim))
        self.bias = nn.Parameter(torch.FloatTensor(4 * opt.hidden_dim))

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def cross_network(self,f0,fn):
        fn_weight = torch.matmul(fn,self.weight)
        fl = f0*fn_weight + self.bias + f0
        x = fl[:,:,0:2*self.opt.hidden_dim]
        y = fl[:,:,2*self.opt.hidden_dim:]
        return x,y

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, pmi_adj, cos_adj, dep_adj = inputs
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        if not self.usebert:
            text_len = torch.sum(text_indices != 0, dim=-1)
            text = self.embed(text_indices)
            text_out = self.text_embed_dropout(text)
            text_out, (_, _) = self.text_lstm(text_out, text_len)
            f0 = torch.cat([text_out, text_out], dim=2)
        else:
            text_len = torch.sum(text_indices != 0, dim=-1)
            text, pool = self.embed(text_indices)
            for ids, s in enumerate(text_len):
                s = s.item() - 2    #句子的单词个数
                pad0 = torch.zeros((text.shape[1]-s, text.shape[2])).to(self.opt.device)    #补上CLS和SEP，还有原来的pad
                text[ids, :, :] = torch.cat((text[ids, 1:s+1, :], pad0), dim=0) #补零恢复形状
            text_out = self.text_embed_dropout(text)
            text_out, (_, _) = self.text_lstm(text_out, text_len - 2)
            f0 = torch.cat([text_out, text_out], dim=2)
        num_layer = self.opt.num_layer
        f_n = None


        for i in range(num_layer):
            if i == 0:
                x_pmi = F.relu(F.softmax(self.gc1(text_out, pmi_adj),dim=-1))
                x_pmi = F.relu(F.softmax(self.gc2(x_pmi, pmi_adj),dim=-1))
                x_cos = F.relu(F.softmax(self.gc3(text_out,cos_adj),dim=-1))
                x_cos = F.relu(F.softmax(self.gc4(x_cos,cos_adj),dim=-1))
                x_s = torch.cat([(x_pmi) ,  (x_cos)],dim=2)
                for j in range(1):
                    x_p, x_c = self.cross_network(f0,x_s)
                    x_s = torch.cat([(x_p) ,  (x_c)],dim=2)

            else:#cross model
                x_p_d = F.relu(self.gc5(self.position_weight(x_p, aspect_double_idx, text_len, aspect_len), dep_adj))
                x_p_d = F.relu(self.gc6(self.position_weight(x_p_d, aspect_double_idx, text_len, aspect_len), dep_adj))
                x_c_d = F.relu(self.gc7(self.position_weight(x_c, aspect_double_idx, text_len, aspect_len), dep_adj))
                x_c_d = F.relu(self.gc8(self.position_weight(x_c_d, aspect_double_idx, text_len, aspect_len), dep_adj))
                f_n = torch.cat([(self.opt.lamda * x_pmi + x_p_d) ,  (self.opt.lamda * x_cos + x_c_d)],dim=2)

        x = self.mask(f_n, aspect_double_idx) #除aspect外置零
        alpha_mat = torch.matmul(x, f0.transpose(1, 2))   #注意力机制
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, f0).squeeze(1)
        output = self.dfc(x)
        return output, text_indices, alpha
