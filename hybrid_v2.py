import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import networkx as nx
import json
import math

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

number_of_thresholds = 10
number_of_sim_users = 5
number_of_products = 200

def getThreshold(t):
	return 5*(t+1)

def getGenreVector(genreString):
	vec = [0]*len(genre)
	for s in genreString.split('|'):
		vec[genre[s]]=1
	return vec

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
	val=np.dot(np.subtract(pivot_table[i], user_data.iloc[i, 0]), np.subtract(pivot_table[j], user_data.iloc[j, 0]))
	return val/(user_data.iloc[i, 1]*user_data.iloc[j, 1])
	# for k in common[0]:
	# 	diff= ( pivot_table[i][k]-user_data.iloc[i, 0] )*( pivot_table[j][k]- user_data.iloc[j, 0])
	# 	val = val+diff
	# return val/ ( max(user_data.iloc[i, 1], 1)*max(user_data.iloc[j, 1], 1) )

#create pandas dataframe


df = pd.read_csv('dataset/ml-1m/ratings.dat', header=None, names=['reviewerID', 'movieID', 'overall', 'unixReviewTime'], sep=':+', engine='python')
df.sort_values('unixReviewTime')

movieDF = pd.read_csv('dataset/ml-1m/movies.dat', header=None, names=['movieID', 'name', 'genre'], sep='::', engine='python')

#create product data
product_data = pd.DataFrame(df.groupby('movieID')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)

top_product = product_data.sort_values('count').tail(number_of_products).index.values
df = df[df.movieID.isin(top_product)]

split_time = df['unixReviewTime'].quantile([.75])[0.75]

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

pivoted_after = after.pivot(index='reviewerID', columns='movieID', values='overall').fillna(0)

#convert before dataframe to numpy array
numpy_array = before.as_matrix(['reviewerID', 'movieID', 'overall'])

#create product purchase matrix
rows, row_pos = np.unique(numpy_array[:, 0], return_inverse=True)
cols, col_pos = np.unique(numpy_array[:, 1], return_inverse=True)

pivot_table = np.zeros((len(rows), len(cols)), dtype=numpy_array.dtype)
pivot_table[row_pos, col_pos] = numpy_array[:, 2]
dense_table = pivot_table

result_precision = np.zeros((number_of_thresholds), dtype=np.float64);
result_recall = np.zeros((number_of_thresholds), dtype=np.float64);
result_f_score = np.zeros((number_of_thresholds), dtype=np.float64);

for target in range(len(accepted_users)):		
	X = []
	Y = []
	for x in np.nonzero(pivot_table[target])[0]:
		# print movieDF[movieDF.movieID==cols[x]].iloc[0, 2]
		X.append(getGenreVector(movieDF[movieDF.movieID==cols[x]].iloc[0, 2]))
		Y.append(pivot_table[target][x])
	classes = len(set(Y))
	if(classes>1):
		clf = svm.SVC()
		clf.fit(X, Y)

		for x in np.where(pivot_table[target]==0)[0]:
			vec = getGenreVector(movieDF[movieDF.movieID==cols[x]].iloc[0, 2])
			prediction = clf.predict([vec])
			dense_table[target][x] = prediction[0]

	else:
		for x in np.where(pivot_table[target]==0)[0]:
			dense_table[target][x] = Y[0]

for target in range(len(accepted_users)):
	print "target :", target
	sim = np.zeros((len(accepted_users)), dtype=np.float64)
	for x in range(len(accepted_users)):
		sim[x] = calcuate_similarity(dense_table, user_data, product_data, target, x)
	sim_users = np.argpartition(sim, -number_of_sim_users)[-number_of_sim_users:]
	purchase_record = set([])
	for u in sim_users:
		for x in np.nonzero(pivot_table[u])[0]:
			purchase_record.add(x)

	purchase_record = list(purchase_record)
	recommend_prob = np.zeros((len(purchase_record)), dtype=np.float64)
	weight_sum = np.zeros((len(purchase_record)), dtype=np.float64)
	for i in range(len(purchase_record)):
		for u in sim_users:
			if(dense_table[u][purchase_record[i]]!=0):
				recommend_prob[i] += sim[u]*(dense_table[u][purchase_record[i]]-user_data.iloc[u, 0])
				weight_sum[i] += sim[u]

	for i in range(len(purchase_record)):
		recommend_prob[i] = user_data.iloc[target, 0] + recommend_prob[i]/weight_sum[i]

	min_rating = recommend_prob.min()
	max_rating = recommend_prob.max()
	recommend_prob = np.subtract(recommend_prob, min_rating)
	recommend_prob = np.true_divide(recommend_prob, (max_rating - min_rating))

	# print recommend_prob.mean(), recommend_prob.min(), recommend_prob.max()
	for t in range(number_of_thresholds):
		threshold = getThreshold(t)
		recommendation_list = np.argpartition(recommend_prob, -threshold)[-threshold:]
		count=0
		after_purchased_count = len(after[after.reviewerID==rows[target]])
		for x in recommendation_list:
			if(len(after[(after.reviewerID==rows[target]) & (after.movieID==cols[purchase_record[x]])])>0):
				count+=1
		if len(recommendation_list)>0 :
			precision = count*1.0/len(recommendation_list)
			result_precision[t] += precision
			recall = count*1.0/after_purchased_count
			result_recall[t] += recall
			if((precision+recall)>0):
				f = 2.0*precision*recall/(precision+recall)
				result_f_score[t] += f
		print count, len(recommendation_list), after_purchased_count
		# print "count :", count
		# print "precision :", precision

	if target>=50:
		break;

result_precision = np.true_divide(result_precision, 51)
result_recall = np.true_divide(result_recall, 51)
result_f_score = np.true_divide(result_recall, 51)

f = open('hybrid_wrt_list_len.csv', 'w+')
for i in range(number_of_thresholds):
	s = str(getThreshold(i))+", "+str(result_precision[i])+", "+str(result_recall[i])+", "+str(result_f_score[i])+"\n"
	f.write(s)
f.close()

plt.plot([getThreshold(i) for i in range(number_of_thresholds)], result_precision)
plt.axis([5.0, 50.0, 0.0, 1.0])
plt.show()



