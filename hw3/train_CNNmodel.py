# %load CNN_model_1.py
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping
import sys
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt

def span(num, X):
    Y = np.zeros((X.shape[0], num))
    index = 0
    for i in X:
        Y[index][i] = 1
        index += 1

    return Y

def normalize(X):
    X = (X - X.mean()) / X.std()
    return X

train_data = pd.read_csv(sys.argv[1])
#print(train_data)

train_y = train_data['label']
train_y = np.array(train_y, int)        # one row

#train_y = train_y.reshape(-1,1)         # one column
#train_y = train_data['label'].tolist()

train_y = span(7, train_y)
#print(train_y)

train_x = np.array([row.split(' ') for row in train_data['feature'].tolist()], dtype = float)
#train_x = map(float, train_x)
#print(train_x)
train_x = normalize(train_x)
#print(train_x)
#print('building model')
train_vaild_x = train_x[:2000,:]
train_x = train_x[2000:,:]
train_vaild_y = train_y[:2000,:]
train_y = train_y[2000:,:]

#print(train_x , train_y)

train_x = train_x.reshape(-1,48,48,1)
train_vaild_x = train_vaild_x.reshape(-1,48,48,1)
#print(train_x)
weight_decay = 0.00001

model = Sequential()

model.add(Conv2D(64, kernel_size = (5,5), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay), kernel_initializer='glorot_normal', input_shape = (48,48,1)))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, kernel_size=(3,3), padding='same',kernel_regularizer = regularizers.l2(weight_decay), kernel_initializer='glorot_normal' ))
model.add(PReLU())
model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(512, kernel_size=(3,3), padding='same',kernel_regularizer = regularizers.l2(weight_decay), kernel_initializer='glorot_normal' ))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

#model.add(Conv2D(256, kernel_size=(3,3), padding='same',kernel_regularizer = regularizers.l2(weight_decay) ))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.3))

model.add(Flatten())
#model.add(Dense(units = 1024))
#model.add(PReLU())
#model.add(Dropout(0.35))
model.add(Dense(units = 512))
model.add(PReLU())
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(units = 512))
model.add(PReLU())
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(units = 7))
model.add(Activation('softmax'))

#print('compiling')

model.summary()

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center = False,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        vertical_flip = False
)

datagen.fit(train_x)

opt = keras.optimizers.Adam(lr = 0.00025, decay = 1e-6)

model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

#print('fitting')
early_stop = EarlyStopping(monitor='val_acc', patience=7, verbose=1)

model.fit_generator(datagen.flow(train_x, train_y, batch_size = 128), epochs = 75, verbose = 1, validation_data = (train_vaild_x , train_vaild_y), workers = 5)

#print('saving')

model.save('CNN_test_28.h5')

scores = model.evaluate(train_vaild_x, train_vaild_y, batch_size = 128, verbose = 1 )
print('result %.3f , loss %.3f' %(scores[1]*100, scores[0]))
