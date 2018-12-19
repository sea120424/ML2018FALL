import pandas as pd
import numpy as np
import re
import csv
import jieba
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import sys

jieba.set_dictionary(sys.argv[4])

def load_x_data(data_path):
    with open(data_path, "r", encoding = 'utf-8') as fp:    
        article = fp.readlines()
        sentance = [re.sub('^[0-9]+,','',s) for s in article[1:]]
    return sentance

def load_y_data(label_path):
    label = pd.read_csv(label_path)['label']
    label = np.array(label)

    return label


train_x = load_x_data(sys.argv[2])
test_x = load_x_data(sys.argv[4])
train_y = load_y_data(sys.argv[3])
print('end load')
#print(type(train_x),type(test_x))
all_data = []
all_data.extend(train_x)
all_data.extend(test_x)
#max_length = max([ len(line) for line in all_data])
#print(max_length)

#print(train_x)
sent = [list(jieba.cut(s, cut_all = False)) for s in all_data]

sent_train = [list(jieba.cut(s, cut_all = False)) for s in train_x]
print('end jibea')

emb_dim = 250

#emb_model = Word2Vec(sent, size = emb_dim, iter = 5, min_count = 5)
#emb_model.save('w2v_3.bin')

emb_model = Word2Vec.load('w2v_3.bin')
num_words = len(emb_model.wv.vocab) + 1
print('end load emb_model')
print(num_words)

emb_matrix = np.zeros((num_words, emb_dim), dtype = float)
for i in range(num_words - 1):
    v = emb_model.wv[emb_model.wv.index2word[i]]
    emb_matrix[i+1] = v

train_sequences = []

for i, s in enumerate(sent_train):
    temp = []
    for w in s:
        if w in emb_model.wv.vocab:
            toks = emb_model.wv.vocab[w].index + 1
            temp.append(toks)
    train_sequences.append(temp)

print('end padding')   
max_length = 80
train_sequences = pad_sequences(train_sequences, maxlen = max_length)
train_val_x = train_sequences[:10000]
train_x = train_sequences[10000:]
train_val_y = train_y[:10000]
train_y = train_y[10000:]
print(train_sequences.shape)
print(train_x)
print(train_x.shape)
print(len(train_y))
model = Sequential()
model.add(Embedding(num_words, emb_dim, weights = [emb_matrix], input_length = max_length, trainable = False))
model.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

print('end building')

adam = Adam(lr = 0.001, decay = 1e-6, clipvalue = 0.5)
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

csv_logger = CSVLogger('training.log')
checkpoint = ModelCheckpoint(filepath = 'best_4.h5',verbose = 1,save_best_only = True, monitor = 'val_acc', mode = 'max')
earlystopping = EarlyStopping(monitor = 'val_acc', patience = 6, verbose = 1, mode = 'max')

#history = LossHistory()

model.fit(train_x, train_y, validation_data = (train_val_x, train_val_y), epochs = 8, batch_size = 512, callbacks = [earlystopping, checkpoint, csv_logger])
print('end trainning')
# tmp -> initial
# tmp2 -> maxlength 150 -> 120, epochs 5 -> 8, batch_size = 512 -> 1024, LSTM = 256 -> 512
# tmp3 -> maxlength 120 -> 96, LSTM = 256 -> 512, min_count = 10


#history.loss_plot('epoch')
