import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
import math
from sklearn import preprocessing

number_of_thresholds = 10
number_of_products = 200
number_of_sim_users = 5
alpha = 0.5
step = 5

genre = dict()
genre['Action']=0
genre['Adventure']=1
genre['Animation']=2
genre['Children\'s']=3
genre['Comedy']=4
genre['Crime']=5
genre['Documentary']=6
genre['Drama']=7
genre['Fantasy']=8
genre['Film-Noir']=9
genre['Horror']=10
genre['Musical']=11
genre['Mystery']=12
genre['Romance']=13
genre['Sci-Fi']=14
genre['Thriller']=15
genre['War']=16
genre['Western']=17

genre_reverse = {v: k for k, v in genre.iteritems()}

def getThreshold(t):
	return 5*(t+1)

def getGenreVector(genreString):
	vec = [0]*len(genre)
	for s in genreString.split('|'):
		vec[genre[s]]=1
	return pd.Series(vec)

def calculate_trust(row, max_freq):
	sigma = 0.5;
	return (sigma*row['mean'])/5+((1-sigma)*row['count'])/max_freq

def normalizeVector(row):
	genre_vector = row.as_matrix()
	rms = np.sqrt(np.sum(np.square(genre_vector)))
	return pd.Series(np.true_divide(genre_vector, rms))

#Read ratings data
df = pd.read_csv('dataset/ml-1m/ratings.dat', 
				header=None, 
				names=['reviewerID', 'productID', 'overall', 'unixReviewTime'], 
				sep=':+', 
				engine='python',)
df.sort_values('unixReviewTime')

#Read movie data
movieDF = pd.read_csv('dataset/ml-1m/movies.dat', 
			header=None, 
			names=['productID', 'name', 'genre'], 
			sep='::', 
			engine='python')

print "Finished reading data"

#create product data and filter top products
product_data = pd.DataFrame(df.groupby('productID')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)
top_product = product_data.sort_values('count').tail(number_of_products).index.values
df = df[df.productID.isin(top_product)]
print "Created product data"

genre_list = sorted(genre.keys())
newcols = movieDF['genre'].apply(getGenreVector)
newcols.columns = genre_list
movieDF = movieDF.join(newcols)

split_time = df['unixReviewTime'].quantile([.75])[0.75]
after = df[df.unixReviewTime>split_time]
before = df[df.unixReviewTime<=split_time]

user_before = before.reviewerID.unique().tolist()
user_after = after.reviewerID.unique().tolist()
common_users = set(user_before).intersection(set(user_after))
before = before[before.reviewerID.isin(common_users)]
before = pd.merge(before, movieDF, on='productID')

#create user data
user_data = pd.DataFrame(before.groupby('reviewerID')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)
user_vector = pd.DataFrame(before.groupby('reviewerID')[genre_list].sum()).fillna(1)
user_vector = user_vector.apply(normalizeVector, axis=1)
user_vector = user_vector.rename(columns=genre_reverse)
print "Created user data"

user_data = user_data[user_data['count'] > 10]
accepted_users = user_data.index.values
print "no. of accepted users :", len(accepted_users)
before = before[before.reviewerID.isin(accepted_users)]

#convert before dataframe to numpy array
numpy_array = before.as_matrix(['reviewerID', 'productID', 'overall'])

#create product purchase matrix
pivoted_after = after.pivot(index='reviewerID', columns='productID', values='overall').fillna(0)
rows, row_pos = np.unique(numpy_array[:, 0], return_inverse=True)
cols, col_pos = np.unique(numpy_array[:, 1], return_inverse=True)
pivot_table = np.zeros((len(rows), len(cols)), dtype=numpy_array.dtype)
pivot_table[row_pos, col_pos] = numpy_array[:, 2]

#calculate trust
max_fre = product_data['count'].max()
product_data['trust'] = product_data.apply (lambda row: calculate_trust (row, max_fre),axis=1)

print "No.of products :", len(product_data);

result_precision = np.zeros((number_of_thresholds), dtype=np.float64);
result_recall = np.zeros((number_of_thresholds), dtype=np.float64);
result_f_score = np.zeros((number_of_thresholds), dtype=np.float64);

# for target in range(len(accepted_users)):
# 	print "target :", target
# 	sim = np.array([calcuate_similarity(pivot_table, user_data, product_data, target, x) for x in range(len(accepted_users))])	
# 	sim_users = np.argpartition(sim, -number_of_sim_users)[-number_of_sim_users:]
# 	sim_users = np.append(sim_users, [target])

# 	purchase_count = {}
# 	for u in sim_users:
# 		for x in np.nonzero(pivot_table[u])[0]:
# 			if(x in purchase_count):
# 				purchase_count[x]+=1
# 			else:
# 				purchase_count[x]=1

# 	transition_matrix=np.zeros((len(purchase_count),len(purchase_count)),dtype=np.float64 )
# 	corelation_matrix=np.zeros((len(purchase_count),len(purchase_count)),dtype=np.float64 )
# 	purchase_record = purchase_count.keys()
# 	print "No. of candidate products : ", len(purchase_record)

# 	for u in sim_users:
# 		user_purchase = pd.DataFrame(before[before['reviewerID']==rows[u]].sort_values('unixReviewTime'))
# 		purchase_transition = user_purchase['productID'].tolist()
# 		for i in range(len(purchase_transition)-1):
# 			x = np.where(cols == purchase_transition[i])[0][0]
# 			y = np.where(cols == purchase_transition[i+1])[0][0]
# 			transition_matrix[purchase_record.index(x)][purchase_record.index(y)]+=1

# 	for i in range(len(transition_matrix)):
# 		transition_matrix[i] = transition_matrix[i]/purchase_count[purchase_record[i]]

# 	for group in before[before['productID'].isin([cols[i] for i in purchase_record])].groupby('reviewerID'):
# 		for x, row_x in group[1].sort_values('unixReviewTime').tail(3).iterrows():
# 			for y, row_y in group[1].sort_values('unixReviewTime').tail(3).iterrows():
# 				if x!=y:
# 					a = np.where(cols == row_x['productID'])[0][0]
# 					b = np.where(cols == row_y['productID'])[0][0]
# 					corelation_matrix[purchase_record.index(a)][purchase_record.index(b)]+=1
# 					corelation_matrix[purchase_record.index(b)][purchase_record.index(a)]+=1

# 	corelation_matrix = np.reciprocal(np.add(1, np.exp(np.negative(corelation_matrix))))

# 	p = 0.5
# 	transfer_matrix = p*transition_matrix + (1-p)*corelation_matrix;
# 	last_three_purchase = before[before.reviewerID==rows[target]].tail(3)
# 	recent_products = [purchase_record.index(np.where(cols == x)[0][0]) for x in last_three_purchase['productID'].tolist()]

# 	recommend_prob = np.zeros((len(purchase_record)), dtype=np.float64)

# 	for x in recent_products:
# 		recommend_prob = recommend_prob + transfer_matrix[x]

# 	recommend_prob = np.true_divide(recommend_prob, len(recent_products))

# 	#include trust
# 	for x in range(len(purchase_record)):
# 		recommend_prob[x] = alpha*recommend_prob[x] + (1 - alpha)*product_data.iloc[purchase_record[x], 3]
# 	# print recommend_prob.mean(), recommend_prob.min(), recommend_prob.max()
# 	for t in range(number_of_thresholds):
# 		threshold = getThreshold(t)
# 		recommendation_list = np.argpartition(recommend_prob, -threshold)[-threshold:]
# 		count=0;
# 		after_purchased_count = len(after[after.reviewerID==rows[target]])
# 		for x in recommendation_list:
# 			if(len(after[(after.reviewerID==rows[target]) & (after.productID==cols[purchase_record[x]])])>0):
# 				count+=1
# 		if len(recommendation_list)>0 :
# 			precision = count*1.0/len(recommendation_list)
# 			result_precision[t] += precision
# 			recall = count*1.0/after_purchased_count
# 			result_recall[t] += recall
# 			if((precision+recall)>0):
# 				f = 2.0*precision*recall/(precision+recall)
# 				result_f_score[t] += f
# 	if target>=50:
# 		break;

# result_precision = np.true_divide(result_precision, 51)
# result_recall = np.true_divide(result_recall, 51)
# result_f_score = np.true_divide(result_recall, 51)

# f = open('with_trust_5_wrt_list_len.csv', 'w+')
# for i in range(number_of_thresholds):
# 	s = str(getThreshold(i))+", "+str(result_precision[i])+", "+str(result_recall[i])+", "+str(result_f_score[i])+"\n"
# 	f.write(s)
# f.close()

# plt.plot([getThreshold(i) for i in range(number_of_thresholds)], result_precision)
# plt.axis([5.0, 50.0, 0.0, 1.0])
# plt.show()