import numpy as np
import pandas as pd
from keras.models import load_model
import csv
import sys

def normalize(X):
    X = (X - X.mean()) / X.std()
    return X

test_data = pd.read_csv(sys.argv[1])
test_X = np.array([row.split(' ') for row in test_data['feature'].tolist()] ,dtype = float)
test_X = normalize(test_X)
test_X = test_X.reshape(-1, 48, 48, 1)

new_model = load_model('CNN_test_28.h5')

ans = new_model.predict(test_X)

with open(sys.argv[2], 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(['id', 'label'])
    index = 0
    for i in ans:
        writer.writerow([index, str(np.argmax(i))])
        index += 1
    

#print(ans)
'''
k = 0
print('id,label')
for i in ans:
    print( str(k) + ',' + str(np.argmax(i)))
    k += 1
'''
