import pandas as pd
import numpy as np
from collections import defaultdict
import spacy
import pickle
from data_utils import *
from nltk.corpus import stopwords

def co_occurrence(sentences, window_size):
    d = defaultdict(int)
    vocab = set()

    for text in sentences:
        nlp = spacy.load('en_core_web_sm')
        text = nlp(text).text
        text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)
            next_token = text[i+1 : i+1+window_size]
            #print(next_token)
            for t in next_token:
                key = tuple( sorted([t, token]) )
                d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    # for ind in range(len(vocab)):
    #     df.iloc[ind, ind] = df.iloc[ind, ind] + 1
    return df

def pmi(df, positive=True):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total #每个单词与不同词的共现次数除以单词所有共现次数的和
    df = df / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df

def stopword():
    #得到所有的停用词
    stop_words = []
    for w in ['-s', '-ly', '</s>', 's', '$', "'", '+' ,'*', ]:
        stop_words.append(w)
    return stop_words

def pmi_matrix(text):
    nlp = spacy.load('en_core_web_sm')
    document = nlp(text)
    seq_len = len(text.split())
    with open('./datasets/semeval16/rest16_train_pmi_dict.pkl', 'rb') as f1:
        ppmi_dict = pickle.load(f1)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    for token in document:
        for token2 in document:
            try:
                matrix[token.i][token2.i] = ppmi_dict.loc[token.text, token2.text]
            except:
                pass
    #matrix[matrix<0.3] = 0
    return matrix



# fin = open('datasets/semeval16/restaurant_test.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
# sentences = []
# lines = fin.readlines()
# for i in range(0, len(lines)-3, 3):
#     text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
#     aspect = lines[i + 1].lower().strip()
#     sentence = text_left + ' ' + aspect + ' ' + text_right
#     stop_words = stopword()
#     sentence = ' '.join([s for s in sentence.split() if s not in stop_words])
#     sentences.append(sentence)
#
# sentences = sorted(set(sentences),key=sentences.index)  #去除重复的句子
# df = co_occurrence(sentences, 50)   #根据整个语料构建共现词表
# pmi_dict = pmi(df)
# f = open('./datasets/semeval16/rest16_test_pmi_dict.pkl', 'wb')
# pickle.dump(pmi_dict, f)

all_matrix = []
fin = open('datasets/semeval16/restaurant_train.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
lines = fin.readlines()
for i in range(0, len(lines)-3, 3):
    text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
    aspect = lines[i + 1].lower().strip()
    sentence = text_left + ' ' + aspect + ' ' + text_right
    stop_words = stopword()
    sentence = ' '.join([s for s in sentence.split()]) ##if s not in stop_words])
    print(sentence)
    pmi = pmi_matrix(sentence)
    pmi = pmi/pmi.max()
    pmi[pmi<0.4] = 0
    all_matrix.append(pmi)
print((all_matrix))
f = open('./datasets/semeval16/restaurant_train.raw_pmi.graph', 'wb')
pickle.dump(all_matrix, f)









