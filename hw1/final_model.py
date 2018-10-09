import pandas as pd
import numpy  as np
import scipy
import sys

class Regression():
	def __init__(self):
		pass
	def parameter_init(self, dim):
		self.b = 0
		self.W = np.zeros((dim, 1))
	def feature_scaling(self, X, train=False):
		if train:
			self.min = np.min(X, axis = 0)
			self.max = np.max(X, axis = 0)
		return (X - self.min) / (self.max - self.min)

	def predict(self, X):
		return np.dot(X, self.W) + self.b
	def RMSEloss(self, X ,Y):
		return np.sqrt(np.mean((Y - self.predict(X))** 2) )
	def train(self, X, Y, times=20000, lr=1):
		batch_size = X.shape[0]
		W_dim = X.shape[1]
		self.parameter_init(W_dim)
		X = self.feature_scaling(X, train=True)
		lr_b = 0
		lr_W = np.zeros((W_dim, 1))
		landa = 0
		for time in range(times):
			
			# mse loss
			grad_b = (-np.sum(Y - self.predict(X)) + -np.sum( landa * self.W**2))/batch_size
			#print(grad_b)
	
			grad_W = (-np.dot(X.T, (Y - self.predict(X))  ))/batch_size
			
			# adagrad
			lr_b += grad_b ** 2
			lr_W += grad_W ** 2
			#lr_W += np.dot(grad_W.T, grad_W)			

			#update
			self.b = self.b - lr/np.sqrt(lr_b) * grad_b
			self.W = self.W - lr/np.sqrt(lr_W) * grad_W





data = pd.read_csv('train.csv' , encoding = 'big5')
data.replace('NR','0',inplace = True)
pure_data = data.iloc[:,3:]
data_list = []


#print(pure_data)
for i in range(240):
	temp_data = pure_data.iloc[18*i:18*(i+1),:]
	temp_data = temp_data.reset_index()
	del temp_data['index']
	data_list.append(temp_data)


real_data = pd.concat(data_list, axis=1)
#real_answer = real_data.loc[9,:]
#print(real_data)	
#print(real_answer)







train_list = []
answer_list = []
for i in range(12):
	for j in range(471):
		train_data = real_data.iloc[:,i*240+j:i*240+j+9]
		#train_data.columns = np.array(range(162))
		train_data = train_data.values.reshape(1,-1)
		train_data = pd.DataFrame(data = train_data)
		#train_data = train_data.reshape(1,-1)
		#train_data.columns = pd.DataFrame(train_data,columns=162)
		train_answer = real_data.iloc[9:10,i*240+j+9]
		#train_answer.columns = ['1']
		train_list.append(train_data)
		answer_list.append(train_answer)


drop_list = []
index = 0

'''
for ele in train_list:
	if index < 5:
		print(ele.ix['80'])
		index += 1

for ele in train_list:

	print(ele[80:89])
	for ele_PM25 in ele[80:89]:
		print(ele_PM25)
		if int(ele_PM25) <= 0 or int(ele_PM25) >= 125:
			drop_list.append(index)
			break					
	index += 1

'''
data_x = pd.concat(train_list)
#x = np.array(data_x,float)
data_y = pd.concat(answer_list)
#y = np.array(data_y,float)

x = np.array(data_x,float)
y = np.array(data_y,float)
#print(x)
for ele in x:
	for j in ele[81:90]:
		if j <= 0 or j >= 135:
			drop_list.append(index)
			break		
	index += 1

index = 0
for ele in y:
	if ele <= 0 or ele >= 135:
		drop_list.append(index)
	index += 1

#print(drop_list)
drop_list = list(set(drop_list))
x = pd.DataFrame(data = x)
y = pd.DataFrame(data = y)
x = x.drop(drop_list)
y = y.drop(drop_list)
#print(x)
#print(y)
x = x.values
y = y.values
x = np.array(x,float)
y = np.array(y,float)

#print(x)
#print(y)
# start training
'''
learning_rate = 0.1

w = np.zeros(len(x[0]))
s_grad = np.zeros(len(x[0]))
recrusive = 10000
#print('start_train')
while recrusive:
	tmp = np.dot(x,w)
	loss = y - tmp
	grad = np.dot(x.T,loss)*(-2)
	s_grad += grad**2
	ada = np.sqrt(s_grad)
	w = w - learning_rate * grad / ada
	recrusive -= 1

#print('end')
'''


'''
batch_size = x.shape[0]
W_dim = x.shape[1]
self_b = 0
self_W = np.zeros((W_dim,1))
lr = 0.1
lr_b = 0
lr_W = np.zeros((W_dim, 1))

print(batch_size, W_dim)

for epoch in range(10000):
	# mse loss
	grad_b = -np.sum(y - np.dot(x,self_W) - self_b)/ batch_size
	grad_W = -np.dot(x.T, (y - np.dot(x,self_W) - self_b )) / batch_size
	# adagrad
	print(grad_W)
	lr_b += grad_b ** 2
	lr_W = lr_W + (grad_W ** 2)
	print(lr_W)
	#update
	
	self_b = self_b - lr / np.sqrt(lr_b) * grad_b
	self_W = self_W - lr / np.sqrt(lr_W) * grad_W
'''

A = Regression()
A.train(x,y) 
w = A.W

test_data = pd.read_csv('test.csv',header=None)
test_data = test_data.iloc[:,2:]
test_data.replace('NR','0',inplace = True)

#print(test_data)

test_list = []


for i in range(260):
	temp_x = test_data.iloc[i*18:(i+1)*18,:]
	temp_x = temp_x.values.reshape(1,-1)
	temp_x = pd.DataFrame(data = temp_x)
	test_list.append(temp_x)


test_x = pd.concat(test_list, axis = 0)
test_x = np.array(test_x,float)
#print(test_x)
#print(test_x.shape[0], test_x.shape[1])
index = 0
#drop_list = []

for ele in test_x:
	#ele_index = 1
	for i in range(18):
		ele_index = 9*i+1
		for j in ele[9*i+1:9*(i+1)]:
			if j <= 0 :
				#print(index)
				ele[ele_index] = ele[ele_index-1]
			ele_index += 1
	index += 1



#B = Regression()
test_x = A.feature_scaling(test_x)
y_guess = A.predict(test_x)

print('[')
for i in A.W:
	print(',' + str(i[0]))

print(']')
'''
print('----------')

print(A.b)

'''

'''
print(list(A.min))

print('-------------')

print(list(A.max))
'''

'''
k = 0
print('id,value')
for i in y_guess:
	#print(i)
	print('id_'+str(k) + ',' + str(i[0]) )
	k += 1
'''


