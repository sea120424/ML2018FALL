import pandas as pd
import numpy as np
import sys

def normalization(X_train , X_test):
    X_all = np.concatenate((X_train, X_test))
    mu = (sum(X_all) / X_all.shape[0])
    sigma = np.std(X_all , axis=0)
    mu = np.tile(mu , (X_all.shape[0], 1))
    sigma = np.tile(sigma, (X_all.shape[0], 1))
    X_all_norm = (X_all - mu) / sigma
    X_train = X_all_norm[:X_train.shape[0]]
    X_test = X_all_norm[X_train.shape[0]:]
    return X_train, X_test

def one_hot(data):
    data = np.array(data,float)
    data_temp = data[:,1:3]
    
    #print(data_temp)
    #max_data = data_temp.max(axis = 0)   
    #print(data.shape[1]) 

    create = [2, 6]
    now_column = 0

    for ele in create:
        #print(ele)
        #print('==========================')
        add_array = np.zeros((data_temp.shape[0],ele))
        #print(add_array)
        for index in range(data_temp.shape[0]):
            feature = int(data_temp[index,now_column])
            #print(feature)

            add_array[index,feature-1] = 1

        #print(add_array)
        #print(data.shape[1])
        
        data = np.concatenate((data,add_array), axis = 1)
        #print(data.shape[1])
        now_column += 1
        #print(data)
    data = np.delete(data, [1, 2, 3], axis = 1)
    #print(data)
    return data
        

class logic_regression:
    def __init__(self):
        pass

    def parameter_init(self , dim ):
        self.b = 0
        self.W = np.zeros((dim,1))
    def sigmz(self , z):
        y = 1 / ( 1 + np.exp(-z))
        return np.clip(y,1e-9 , 1-(1e-9))
    def feature_scaling(self, X , train=False):
        if train:
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        return (X - self.min)/(self.max-self.min)
    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        result = self.sigmz(z) + 0.1
        result = np.around(result)

        return result
    def train(self, X, Y):
        batch_size = X.shape[0]
        W_dim = X.shape[1]
        self.parameter_init(W_dim)
        X = self.feature_scaling(X, train=True)
        lr_b = 0
        lr_W = np.zeros((W_dim , 1))
        
        lr = 1
        times = 15000
        for time in range(times):
            
            z = np.dot(X, self.W) + self.b
            y = self.sigmz(z)
            grad_b = -np.sum(Y - y)/batch_size
            grad_W = -np.dot(X.T, (Y - y))/batch_size
            
            lr_b += grad_b ** 2
            lr_W += grad_W ** 2
            #grad_b = 
            self.b = self.b - lr/np.sqrt(lr_b) * grad_b
            self.W = self.W - lr/np.sqrt(lr_W) * grad_W



data_x = pd.read_csv(sys.argv[1])
data_y = pd.read_csv(sys.argv[2])
test_data = pd.read_csv(sys.argv[3])

data_x = one_hot(data_x)
#print(data_x.shape[1], test_data.shape[1])


test_data = one_hot(test_data)




#print(data_x.shape[1], test_data.shape[1])

#data_x = np.array(data_x,float)
#test_data = np.array(test_data,float)

#one_hot
#one_hot_list = [SEX, EDUCATION, MARRIAGE]


# normalize
data_x, test_data = normalization(data_x, test_data)

data_x = pd.DataFrame(data =  data_x)
test_data = pd.DataFrame(data = test_data)


#data_x.drop(columns = ['SEX'])
data_all = data_x.iloc[:17500,:]
answer_all = data_y[:17500]
data_varify = data_x.iloc[17500:,:]
answer_varify = data_y[17500:]

data_all = np.array(data_all,float)
answer_all = np.array(answer_all,int)
data_varify = np.array(data_varify,float)
answer_varify = np.array(answer_varify,int)


A = logic_regression()
A.train(data_all,answer_all)
#print(data_varify)
data_varify = A.feature_scaling(data_varify)
#print(data_varify)
y_guess = A.predict(data_varify)
#print(y_guess)

#print(A.W)
'''
error = 0

for i in range(2500):
    if y_guess[i] != answer_varify[i]:
        error += 1
        #print(answer_varify[i])
print((2500-error)/2500)

#print(data_all)
#print(data_varify)

#test_data = pd.read_csv('test_x.csv')
'''

test_data = np.array(test_data,float)
test_data = A.feature_scaling(test_data)
predict_y = A.predict(test_data)



k = 0
print('id,value')
for i in predict_y:
    print('id_' + str(k) + ',' + str(int(i[0])))
    k += 1   

