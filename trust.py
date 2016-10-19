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
	w_lambda=0.5
	normalize_freq = np.std(product_data['count'].values, axis=0, ddof=0)
	common = (pivot_table[i]*pivot_table[j]).nonzero()
	val=0
	for k in common[0]:
		diff= ( pivot_table[i][k]-user_data.iloc[i, 0] )*( pivot_table[j][k]-user_data.iloc[j, 0] )
		val0= w_lambda/(product_data.iloc[k, 0]**2)+ (1-w_lambda)/( (product_data.iloc[k, 2]/normalize_freq)**2)
		val = val+ math.sqrt(val0) * diff
	return val/ ( max(user_data.iloc[i, 1], 1)*max(user_data.iloc[j, 1], 1) )

#create pandas dataframe

df = getDF('Digital_Music_5.json')
#df = getDF('test_5500.json')
df.drop(['reviewerName', 'helpful', 'reviewText', 'reviewTime', 'summary'], inplace=True, axis=1)
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

# corelation_matrix=np.zeros((20,20),dtype=np.int32 )
# for group in df[df.asin.isin(top_product)].groupby('reviewerID'):
# 	for x, row_x in group[1].iterrows():
# 		for y, row_y in group[1].iterrows():
# 			if x!=y:
# 				a = np.where(top_product == row_x['asin'])[0][0]
# 				b = np.where(top_product == row_y['asin'])[0][0]
# 				corelation_matrix[a][b]+=1
# 				corelation_matrix[b][a]+=1

# print corelation_matrix

#calculate trust
max_fre = product_data['count'].max()
product_data['trust'] = product_data.apply (lambda row: calculate_trust (row, max_fre),axis=1)

target = 22
threshold = 5

for target in range(len(accepted_users)):
	print "target :", target
	sim = np.array([calcuate_similarity(pivot_table, user_data, product_data, target, x) for x in range(len(accepted_users))])	
	sim_users = np.argpartition(sim, -threshold)[-threshold:]
	sim_users = np.append(sim_users, [target])

	purchase_count = {}

	for u in sim_users:
		for x in np.nonzero(pivot_table[u])[0]:
			if(x in purchase_count):
				purchase_count[x]+=1
			else:
				purchase_count[x]=1

	# print purchase_count

	transition_matrix=np.zeros((len(purchase_count),len(purchase_count)),dtype=np.float64 )
	corelation_matrix=np.zeros((len(purchase_count),len(purchase_count)),dtype=np.float64 )
	purchase_record = purchase_count.keys()

	for u in sim_users:
		user_purchase = pd.DataFrame(before[before['reviewerID']==rows[u]].sort_values('unixReviewTime'))
		purchase_transition = user_purchase['asin'].tolist()
		for i in range(len(purchase_transition)-1):
			x = np.where(cols == purchase_transition[i])[0][0]
			y = np.where(cols == purchase_transition[i+1])[0][0]
			transition_matrix[purchase_record.index(x)][purchase_record.index(y)]+=1

	for i in range(len(transition_matrix)):
		transition_matrix[i] = transition_matrix[i]/purchase_count[purchase_record[i]]

	for group in before[before['asin'].isin([cols[i] for i in purchase_record])].groupby('reviewerID'):
		for x, row_x in group[1].sort_values('unixReviewTime').tail(3).iterrows():
			for y, row_y in group[1].sort_values('unixReviewTime').tail(3).iterrows():
				if x!=y:
					a = np.where(cols == row_x['asin'])[0][0]
					b = np.where(cols == row_y['asin'])[0][0]
					corelation_matrix[purchase_record.index(a)][purchase_record.index(b)]+=1
					corelation_matrix[purchase_record.index(b)][purchase_record.index(a)]+=1

	corelation_matrix = np.reciprocal(np.add(1, np.exp(np.negative(corelation_matrix))))

	p = 0.7
	transfer_matrix = p*transition_matrix + (1-p)*corelation_matrix;
	last_three_purchase = before[before.reviewerID==rows[target]].tail(3)
	recent_products = [purchase_record.index(np.where(cols == x)[0][0]) for x in last_three_purchase['asin'].tolist()]

	recommend_prob = np.zeros((len(purchase_record)), dtype=np.float64)

	for x in recent_products:
		recommend_prob = recommend_prob + transfer_matrix[x]

	recommend_prob = np.true_divide(recommend_prob, len(recent_products))

	count=0;
	for x in np.where(recommend_prob>0.2)[0]:
		if(len(after[(after.reviewerID==rows[target]) & (after.asin==cols[purchase_record[x]])])>0):
			count+=1

	print count

