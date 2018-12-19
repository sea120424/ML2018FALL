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
from keras.preprocessing.text import Tokenizer
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


train_x = load_x_data(sys.argv[1])
test_x = load_x_data(sys.argv[3])
train_y = load_y_data(sys.argv[2])

#print(type(train_x),type(test_x))
all_data = []
all_data.extend(train_x)
all_data.extend(test_x)
#max_length = max([ len(line) for line in all_data])
#print(max_length)

#print(train_x)
sent = [list(jieba.cut(s, cut_all = False)) for s in all_data]
sent_train = [list(jieba.cut(s, cut_all = False)) for s in train_x]
#print(sent)

print('doing BOW')
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(sent)
BOW_matrix = tokenizer.texts_to_matrix(sent_train, mode = 'count')
print('matrix')
print(BOW_matrix)

#max_length = 120
#train_sequences = pad_sequences(train_sequences, maxlen = max_length)
train_val_x = BOW_matrix[:10000]
train_x = BOW_matrix[10000:]
train_val_y = train_y[:10000]
train_y = train_y[10000:]
print(BOW_matrix.shape)
print(train_x)
print(train_x.shape)
print(len(train_y))
model = Sequential()
#model.add(Embedding(num_words, emb_dim, weights = [emb_matrix], input_length = max_length, trainable = False))
#model.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3))
model.add(Dense(512, input_shape = (20000,) ,activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

adam = Adam(lr = 0.001, decay = 1e-6, clipvalue = 0.5)
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

csv_logger = CSVLogger('training.log')
checkpoint = ModelCheckpoint(filepath = 'best_BOW_4.h5',verbose = 1,save_best_only = True, monitor = 'val_acc', mode = 'max')
earlystopping = EarlyStopping(monitor = 'val_acc', patience = 6, verbose = 1, mode = 'max')

model.fit(train_x, train_y, validation_data = (train_val_x, train_val_y), epochs = 5, batch_size = 128, callbacks = [earlystopping, checkpoint, csv_logger])

# tmp -> initial
#tmp2 -> maxlength 150 -> 120, epochs 5 -> 8, batch_size = 512 -> 1024

#history.loss_plot('epoch')
