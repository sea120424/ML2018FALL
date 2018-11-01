import pandas as pd
import numpy as np
import sys

def find_model(data):
    mean = np.mean(data, axis = 0)
    det = np.zeros((data.shape[1],data.shape[1]))
    #print(data.shape[1])
    for i in range(data.shape[0]):
        nor = data[i]-mean
        det += (np.transpose([nor]) * nor) / data.shape[0]

    #print(data[0])
    #norm = data[0] - mean
    #print(norm)
    #print(np.transpose([norm]))
    #print((data[0]-mean) * (data[0]-mean).T)
    #det /= data.shape[0]
    #det = np.sum(np.dot((data-mean),(data-mean).T)) / data.shape[0]
    #print(det.shape[0], det.shape[1])
    return mean, det

def Gaussian(mean, det, data):
    inverse = np.linalg.inv(det)
    det_value = np.sqrt(np.linalg.det(det))
    devided = det_value * np.power(2 * np.pi, det.shape[0]/2)
    #print((data-mean))
    return np.exp(-0.5* np.dot(np.dot((data-mean),inverse) ,(data-mean).T)) / devided
 

def deal(A):
    A = A.diagonal()
    A = np.transpose([A])
    return A
'''
def dumping(zero_mean, one_mean, zero_det, one_det):
    print("zero_mean = ")
    print(zero_mean.tolist())
    print("one_mean = ")
    print(one_mean.tolist())
    print("zero_det = ")
    print(zero_det.tolist())
    print("one_det = ")
    print(one_det.tolist())
    print("\n")
'''

data_x = pd.read_csv(sys.argv[1])
data_y = pd.read_csv(sys.argv[2])
test_data = pd.read_csv(sys.argv[3])

data_all = data_x.iloc[:17500,:]
answer_all = data_y[:17500]
data_varify = data_x.iloc[17500:,:]
answer_varify = data_y[17500:]

data_all = np.array(data_all, float)
answer_all = np.array(answer_all, int)
data_varify = np.array(data_varify, float)
answer_varify = np.array(answer_varify, int)

count_one = 0
one_list = []
count_zero = 0
zero_list = []
for i in range(answer_all.shape[0]):
    if answer_all[i] == 0:
        count_zero += 1
        zero_list.append(i)
    elif answer_all[i] == 1:
        count_one += 1
        one_list.append(i)

#print(count_zero, count_one)

count_zero = count_zero / answer_all.shape[0]
count_one = count_one / answer_all.shape[0]

data_one = data_all[one_list,:]
data_zero = data_all[zero_list,:]

mean_one, det_one = find_model(data_one)
mean_zero , det_zero = find_model(data_zero)

#dumping(mean_zero, mean_one, det_zero, det_one)

'''
fun_one = Gaussian(mean_one, det_one, data_varify)
fun_zero = Gaussian(mean_zero, det_zero, data_varify)

#print(fun_one)
fun_one = deal(fun_one)
fun_zero = deal(fun_zero)
#print(fun_one.shape)

prop = fun_one * count_one / (fun_one * count_one + fun_zero * count_zero + 1e-99)

#print(prop)
prop -= 0.2
prop = np.around(prop) 

#print(prop)

error = 0

for i in range(2500):
    if prop[i] != answer_varify[i]:
        error += 1

print((2500 - error)/2500)
'''

test_data = np.array(test_data,float)
test_fun_one = Gaussian(mean_one, det_one, test_data)
test_fun_zero = Gaussian(mean_zero, det_zero, test_data)
test_fun_one = deal(test_fun_one)
test_fun_zero = deal(test_fun_zero)

prop_test = test_fun_one * count_one / (test_fun_one * count_one + test_fun_zero * count_zero + 1e-99)
prop_test -= 0.4
prop_test = np.around(prop_test)
#print(prop_test)
nan_place = np.isnan(prop_test)
prop_test[nan_place] = 0

k = 0
print('id,value')
for i in prop_test:
    print('id_' + str(k) + ',' + str(int(i[0])))
    k += 1


#print( det_one)

    
