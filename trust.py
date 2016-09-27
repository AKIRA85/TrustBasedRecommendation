import pandas as pd 
import numpy as np
import json
import math

def parse(path): 
	f = open(path)
	for line in f:
		yield json.loads(line)

def getDF(path): 
	i = 0 
	df = {} 
	for d in parse(path): 
		df[i] = d 
		i += 1
	return pd.DataFrame.from_dict(df, orient='index') 

#create pandas dataframe

#df = getDF('Digital_Music_5.json')
df = getDF('test_5500.json')

#convert to numpy array
numpy_array = df.as_matrix(['reviewerID', 'asin', 'overall'])

#create product purchase matrix
rows, row_pos = np.unique(numpy_array[:, 0], return_inverse=True)
cols, col_pos = np.unique(numpy_array[:, 1], return_inverse=True)

pivot_table = np.zeros((len(rows), len(cols)), dtype=numpy_array.dtype)
pivot_table[row_pos, col_pos] = numpy_array[:, 2]

#create user data
user_data = pd.DataFrame(df.groupby('reviewerID')['overall'].agg([np.mean, np.std])).fillna(1)

#create product data
product_data = pd.DataFrame(df.groupby('asin')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)

print len(product_data), len(user_data)

mat_similarity = np.zeros((len(user_data),len(user_data)),dtype=np.float64 )
#normalize_freq = np.std(product_data['count'].values, axis=0, ddof=0)
normalize_freq = 10

w_lambda=0.5

for i in range(len(user_data)):
    for j in range(len(user_data)):
        common = (pivot_table[i]*pivot_table[j]).nonzero()
        val=0
        for k in common[0]:
	        diff= ( pivot_table[i][k]-user_data.iloc[i, 0] )*( pivot_table[j][k]-user_data.iloc[j, 0] )
	        val0= w_lambda/(product_data.iloc[k, 0]**2)+ (1-w_lambda)/( (product_data.iloc[k, 2]/normalize_freq)**2)
	        val = val+ math.sqrt(val0) * diff
	        #print('\t***',diff,k,pivot_table[i][k],user_sigma[i][1])
	    #print(i,j,val,user_sigma[i][2],user_sigma[j][2])
        mat_similarity[i][j]= val/ ( max(user_data.iloc[i, 1], 1)*max(user_data.iloc[j, 1], 1) )

np.savetxt('similarity.csv',mat_similarity,delimiter=',')