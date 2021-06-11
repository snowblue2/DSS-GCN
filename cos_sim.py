import numpy as np
import torch.nn as nn
import torch
import pickle
from transformers import BertModel,BertTokenizer

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                #每个word一个id
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words] #根据word2idx重组每句话
        if len(sequence) == 0:
            sequence = [0]
        return sequence





with open('./300_twitter_embedding_matrix.pkl', 'rb') as f:     #特定数据集的glove词嵌入
    embedding_matrix = pickle.load(f)


with open('./twitter_word2idx.pkl', 'rb') as f:   #特定数据集的词表
    word2idx = pickle.load(f)
    tokenizer = Tokenizer(word2idx=word2idx)

#加载glove词嵌入，对句子进行token，然后计算单词间的余弦相似度，构建Cosine similarity图
usebert = 1
if not usebert:
    embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float)) #glvoe矩阵
else:
    embed = BertModel.from_pretrained('./datasets/bert-base-uncased').requires_grad_(False) #BERT
    embed.eval()
all_matrix = []

fin = open('datasets/semeval14/restaurant_test.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
lines = fin.readlines()
for i in range(0, len(lines)-3, 3):
    text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
    aspect = lines[i + 1].lower().strip()
    sentence = text_left + ' ' + aspect + ' ' + text_right
    if not usebert:
        text_indices = tokenizer.text_to_sequence(sentence)
        text_embed = embed(torch.tensor(text_indices))
    else:
        berttokenizer = BertTokenizer.from_pretrained('./datasets/bert-base-uncased')
        text_indices = torch.tensor(berttokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)
        text_bert, pool = embed(text_indices)
        text_embed = text_bert[0, 1:-1, :] #取文本长度

    cossim_matrix = np.zeros((len(text_embed), len(text_embed))).astype('float32')
    for num1 in range(len(text_embed)):
        for num2 in range(len(text_embed)):
            cossim_matrix[num1][num2] = torch.cosine_similarity(text_embed[num1], text_embed[num2], dim=0)
    cossim_matrix[cossim_matrix<0.4] = 0
    all_matrix.append(cossim_matrix)
print(len(all_matrix))

f = open('./datasets/semeval14/restaurant_test.raw_cos.graph_bert', 'wb')
pickle.dump(all_matrix, f)

