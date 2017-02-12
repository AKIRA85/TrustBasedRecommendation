import pandas as pd 
import numpy as np
import json
import math

number_of_thresholds = 11

def getThreshold(t):
	return (t*1.0)/(number_of_thresholds-1)

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


df = pd.read_csv('dataset/ml-1m/ratings.dat', header=None, names=['reviewerID', 'movieID', 'overall', 'unixReviewTime'], sep=':+', engine='python')
# df = getDF('Digital_Music_5.json')
#df = getDF('test_5500.json')
# df.drop(['reviewerName', 'helpful', 'reviewText', 'reviewTime', 'summary'], inplace=True, axis=1)
df.sort_values('unixReviewTime')

#create product data
product_data = pd.DataFrame(df.groupby('movieID')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)

top_product = product_data.sort_values('count').tail(100).index.values
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

number_of_sim_users = 5

result_precision = np.zeros((number_of_thresholds), dtype=np.float64);
result_recall = np.zeros((number_of_thresholds), dtype=np.float64);
result_f_score = np.zeros((number_of_thresholds), dtype=np.float64);

for target in range(len(accepted_users)):
	print "target :", target
	sim = np.zeros((len(accepted_users)), dtype=np.float64)
	for x in range(len(accepted_users)):
		sim[x] = calcuate_similarity(pivot_table, user_data, product_data, target, x)
	sim_users = np.argpartition(sim, -number_of_sim_users)[-number_of_sim_users:]

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
				recommend_prob[i] += sim[u]*(pivot_table[u][purchase_record[i]]-user_data.iloc[u, 0])
				weight_sum[i] += sim[u]

	for i in range(len(purchase_record)):
		recommend_prob[i] = user_data.iloc[target, 0] + recommend_prob[i]/weight_sum[i]
	min_rating = recommend_prob.min()
	max_rating = recommend_prob.max()
	recommend_prob = np.subtract(recommend_prob, min_rating)
	recommend_prob = np.true_divide(recommend_prob, (max_rating - min_rating))

	for t in range(number_of_thresholds):
		threshold = getThreshold(t)
		recommendation_list = np.where(recommend_prob>threshold)[0]

		count=0;
		after_purchased_count = len(after[after.reviewerID==rows[target]])
		for x in recommendation_list:
			if(len(after[(after.reviewerID==rows[target]) & (after.movieID==cols[purchase_record[x]])])>0):
				count+=1
		# print t, count, len(recommendation_list)
		if len(recommendation_list)>0 :
			precision = count*1.0/len(recommendation_list)
			result_precision[t] += precision
			recall = count*1.0/after_purchased_count
			result_recall[t] += recall
			print count, after_purchased_count, len(recommendation_list)
			if count>after_purchased_count:
				print "recall"
			if((precision+recall)>0):
				f = 2.0*precision*recall/(precision+recall)
				result_f_score[t] += f
		# print "count :", count
		# print "precision :", precision
	print result_recall

	if target>=50:
		break;

result_precision = np.true_divide(result_precision, 51)
result_recall = np.true_divide(result_recall, 51)
result_f_score = np.true_divide(result_recall, 51)

f = open('collab.csv', 'w+')
for i in range(number_of_thresholds):
	s = str(getThreshold(i))+", "+str(result_precision[i])+", "+str(result_recall[i])+", "+str(result_f_score[i])+"\n"
	f.write(s)
f.close()




