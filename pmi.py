import pandas as pd
import numpy as np
from collections import defaultdict
import spacy
import pickle
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
    expected = np.outer(row_totals, col_totals) / total
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
    for w in ['-s', '-ly', '</s>', 's', '$', "'", '+', "-" ,'*', '/']:
        stop_words.append(w)
    return stop_words

def pmi_matrix(text):
    nlp = spacy.load('en_core_web_sm')
    document = nlp(text)
    seq_len = len(text.split())
    with open('./data/pmi_dict.pkl', 'rb') as f1:
        ppmi_dict = pickle.load(f1)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    for token in document:
        for token2 in document:
            matrix[token.i][token2.i] = ppmi_dict.loc[token.text, token2.text]

    return matrix


ss = ['Text representation learning is the first and essential step for the text classification problem.']
res = pmi_matrix(ss[0])
print(res)




# fin = open('datasets/semeval16/restaurant_train.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
# sentences = []
# lines = fin.readlines()
# for i in range(0, len(lines), 3):
#     text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
#     aspect = lines[i + 1].lower().strip()
#     sentence = text_left + ' ' + aspect + ' ' + text_right
#     stop_words = stopword()
#     sentence = ' '.join([s for s in sentence.split() if s not in stop_words])
#     sentences.append(sentence)
#
# sentences = sorted(set(sentences),key=sentences.index)
# df = co_occurrence(sentences, 50)
# print(df)
# ppmi = pmi(df, positive=True)
# #a = ppmi.iloc[300].sort_values()
# #b = ppmi.sort_values(by="but", ascending = True)
# #c = ppmi.sort_values(by="but", ascending = False)['but']
# print(ppmi['the']['food'])
#
# f = open('./data/pmi_dict.pkl', 'wb')
# pickle.dump(ppmi, f)



