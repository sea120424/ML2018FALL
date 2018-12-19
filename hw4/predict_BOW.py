from keras.models import load_model
import pandas as pd
import numpy as np
import re
import csv
#import jieba
from keras.preprocessing.text import Tokenizer

def load_x_data(data_path):
    with open(data_path, "r", encoding = 'utf-8') as fp:    
        article = fp.readlines()
        sentance = [re.sub('^[0-9]+,','',s) for s in article[1:]]
    return sentance

def load_y_data(label_path):
    label = pd.read_csv(label_path)['label']
    label = np.array(label)

    return label

train_x = load_x_data('train_x.csv')
test_x = load_x_data('test_x.csv')
print('end load')
all_data = []
all_data.extend(train_x)
all_data.extend(test_x)
#test_x = ['在說別人白痴之前,先想想自己','在說別人之前先想想自己,白痴']
sent = [list(jieba.cut(s, cut_all = False)) for s in all_data]
sent_test = [list(jieba.cut(s, cut_all = False)) for s in test_x]
print('end jieba')
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(sent)
print('end fit')
BOW_test_matrix = tokenizer.texts_to_matrix(sent_test, mode = 'count')
print('end BOW')
'''
test_sequences = []

for i, s in enumerate(sent_test):
    temp = []
    for w in s:
        if w in emb_model.wv.vocab:
            toks = emb_model.wv.vocab[w].index + 1
            temp.append(toks)
    test_sequences.append(temp)
'''

#test_sequences = pad_sequences(test_sequences, maxlen = 120)

model = load_model('best_BOW_4.h5')
pred = model.predict(BOW_test_matrix)
print(pred)

with open('tmp_BOW_4.csv','w',newline = '') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',' , )
    writer.writerow(['id', 'label'])
    index = 0
    for i in pred:
        writer.writerow([index, int(np.around(i))])
        index += 1
        
