from keras.models import load_model
import pandas as pd
import numpy as np
import re
import csv
import jieba
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys

jieba.set_dictionary(sys.argv[2])

def load_x_data(data_path):
    with open(data_path, "r", encoding = 'utf-8') as fp:    
        article = fp.readlines()
        sentance = [re.sub('^[0-9]+,','',s) for s in article[1:]]
    return sentance

def load_y_data(label_path):
    label = pd.read_csv(label_path)['label']
    label = np.array(label)

    return label

test_x = load_x_data(sys.argv[1])
#test_x = ['在說別人白痴之前,先想想自己','在說別人之前先想想自己,白痴']
print('end load')
sent_test = [list(jieba.cut(s, cut_all = False)) for s in test_x]
print('end jieba')
test_sequences = []

emb_dim = 250
emb_model = Word2Vec.load('w2v_3.bin')
num_words = len(emb_model.wv.vocab) + 1

for i, s in enumerate(sent_test):
    temp = []
    for w in s:
        if w in emb_model.wv.vocab:
            toks = emb_model.wv.vocab[w].index + 1
            temp.append(toks)
    test_sequences.append(temp)
print('end matrix')
test_sequences = pad_sequences(test_sequences, maxlen = 80)
model = load_model('best_4.h5')
pred = model.predict(test_sequences)
print(pred)

with open(sys.argv[3],'w',newline = '') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',' , )
    writer.writerow(['id', 'label'])
    index = 0
    for i in pred:
        writer.writerow([index, int(np.around(i))])
        index += 1

