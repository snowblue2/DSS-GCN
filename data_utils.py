import os
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BasicTokenizer

def load_pki(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():    #以空格分开，匹配到在表里的字符时，将后面的向量放入word_vec
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


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

class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer, post_vocab, usebert=0):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        fin = open(fname+'_pmi.graph', 'rb')
        pmi_graph = pickle.load(fin)
        fin.close()

        fin = open(fname+'_cos.graph_bert', 'rb')
        cos_graph = pickle.load(fin)
        fin.close()

        fin = open(fname+'_dep.graph', 'rb')
        dep_graph = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines)-3, 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            sentence = text_left+' '+aspect+' '+text_right
            if not usebert:
                text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
                context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
                aspect_indices = tokenizer.text_to_sequence(aspect)
                left_indices = tokenizer.text_to_sequence(text_left)
            else:
                berttokenizer = BertTokenizer.from_pretrained('./datasets/bert-base-uncased')
                text_indices = (berttokenizer.encode(text_left + " " + aspect + " " + text_right, add_special_tokens=True))
                context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
                aspect_indices = tokenizer.text_to_sequence(aspect)
                left_indices = tokenizer.text_to_sequence(text_left)
                # context_indices = (berttokenizer.encode(text_left + text_right, add_special_tokens=False))
                # aspect_indices = (berttokenizer.encode(aspect, add_special_tokens=False))
                # left_indices = (berttokenizer.encode(text_left, add_special_tokens=False))

            #position = list(range(-left_len,0)) + [0]*aspect_len + list(range(1,right_len + 1))
            # post_emb = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in position]
            polarity = int(polarity)+1
            pmi_graph1 = pmi_graph[i//3]
            cos_graph1 = cos_graph[i//3]
            dep_graph1 = dep_graph[i]
            data = {
                'context':sentence,
                'aspect':aspect,
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'pmi_graph': pmi_graph1,
                'cos_graph':cos_graph1,
                'dep_graph': dep_graph1,

            }
            all_data.append(data)
        return all_data

    def __init__(self,  dataset='twitter', embed_dim=300, post_vocab=None, usebert=0):
        print("preparing {0} dataset ...".format(dataset))
        self.usebert = usebert
        fname = {
            'twitter': {
                'train': './datasets/twitter/twitter_train.raw',
                'test': './datasets/twitter/twitter_test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },

        }

        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])  # 所有的text去生成word2idx

        if os.path.exists(dataset+'_word2idx.pkl'):     #加载特定数据集的word2idx
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)

        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)

        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer, post_vocab, usebert))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer,post_vocab, usebert))




