import pandas as pd 
import numpy as np
import networkx as nx
import json
import math

def calculate_trust(row, max_freq):
	sigma = 0.5;
	return sigma*row['mean']+((1-sigma)*row['count'])/max_freq

def logic_function(x):
	return 1/(1+math.exp(-x))

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

def calcuate_similarity(pivot_table, user_data, product_data, i, j):
	if i==j:
		return 0
	common = (pivot_table[i]*pivot_table[j]).nonzero()
	val=0
	for k in common[0]:
		diff= ( pivot_table[i][k]-user_data.iloc[i, 0] )*( pivot_table[j][k]-user_data.iloc[j, 0] )
		val = val+diff
	return val/ ( max(user_data.iloc[i, 1], 1)*max(user_data.iloc[j, 1], 1) )

#create pandas dataframe

df = pd.read_csv('rating_Electronics.csv')
# df = getDF('Digital_Music_5.json')
# #df = getDF('test_5500.json')
# df.drop(['reviewerName', 'helpful', 'reviewText', 'reviewTime', 'summary'], inplace=True, axis=1)
df.sort_values('unixReviewTime')

#create product data
product_data = pd.DataFrame(df.groupby('asin')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)

# top_product = product_data.sort_values('count').tail(20).index.values
# df = df[df.asin.isin(top_product)]

split_time = 1245000000

after = df[df.unixReviewTime>split_time]
before = df[df.unixReviewTime<=split_time]

user_before = before.reviewerID.unique().tolist()
user_after = after.reviewerID.unique().tolist()

common_users = set(user_before).intersection(set(user_after))

before = before[before.reviewerID.isin(common_users)]

#create user data
user_data = pd.DataFrame(before.groupby('reviewerID')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)
user_data = user_data[user_data['count'] > 4]
accepted_users = user_data.index.values
before = before[before.reviewerID.isin(accepted_users)]

pivoted_after = after.pivot(index='reviewerID', columns='asin', values='overall').fillna(0)

#convert before dataframe to numpy array
numpy_array = before.as_matrix(['reviewerID', 'asin', 'overall'])

#create product purchase matrix
rows, row_pos = np.unique(numpy_array[:, 0], return_inverse=True)
cols, col_pos = np.unique(numpy_array[:, 1], return_inverse=True)

pivot_table = np.zeros((len(rows), len(cols)), dtype=numpy_array.dtype)
pivot_table[row_pos, col_pos] = numpy_array[:, 2]

#calculate trust
max_fre = product_data['count'].max()
product_data['trust'] = product_data.apply (lambda row: calculate_trust (row, max_fre),axis=1)

target = 22
threshold = 5

for target in range(len(accepted_users)):
	print "target :", target
	sim = np.array([calcuate_similarity(pivot_table, user_data, product_data, target, x) for x in range(len(accepted_users))])
	sim_users = np.argpartition(sim, -threshold)[-threshold:]

	purchase_count = {}

	for u in sim_users:
		for x in np.nonzero(pivot_table[u])[0]:
			if(x in purchase_count):
				purchase_count[x]+=1
			else:
				purchase_count[x]=1

	# print purchase_count

	purchase_record = purchase_count.keys()

	recommend_prob = np.zeros((len(purchase_record)), dtype=np.float64)
	weight_sum = np.zeros((len(purchase_record)), dtype=np.float64)

	for i in range(len(purchase_record)):
		for u in sim_users:
			if(pivot_table[u][purchase_record[i]]!=0):
				recommend_prob[i] += sim[u]*(pivot_table[u][purchase_record[i]]-user_data.iloc[i, 0])
				weight_sum += sim[u]

	for i in range(len(purchase_record)):
		recommend_prob[i] = user_data.iloc[target, 0] + recommend_prob[i]/weight_sum[i]

	count=0;
	print len(np.where(recommend_prob>0)[0])
	for x in np.where(recommend_prob>0)[0]:
		if(len(after[(after.reviewerID==rows[target]) & (after.asin==cols[purchase_record[x]])])>0):
			count+=1

	print count

