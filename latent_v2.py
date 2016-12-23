import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
import math

number_of_sim_users = 5
number_of_thresholds = 10
seconds_in_a_month = 60*60*24*30

def getThreshold(t):
	return 5*(t+1)

def calculate_trust(row, max_freq):
	sigma = 0.5;
	return (sigma*row['mean'])/5+((1-sigma)*row['count'])/max_freq

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

def calcuate_latent_similarity(row1, row2):
	w = 0.5
	if row1.iloc[0, 0]==row2.iloc[0, 0]:
		return 0
	sim_age = math.exp(abs(row1['age'].iloc[0]-row2['age'].iloc[0])/(-2))
	sim_g =0 
	if(row1['gender'].iloc[0]==row2['gender'].iloc[0]):
		sim_g = 1
	return w*sim_age+(1-w)*sim_g

#create pandas dataframe

# df = pd.read_csv('ratings_Electronics.csv', header=None, names=['reviewerID', 'movieID', 'overall', 'unixReviewTime'], sep=',', engine='python')

df = pd.read_csv('dataset/ml-1m/ratings.dat', header=None, names=['reviewerID', 'movieID', 'overall', 'unixReviewTime'], sep=':+', engine='python')
df.sort_values('unixReviewTime')
first_purchase_time = df['unixReviewTime'].min()

user_df = pd.read_csv('dataset/ml-1m/users.dat', 
							header=None, 
							names=['reviewerID', 'gender', 'age', 'occupation', 'zip_code'], 
							usecols=['reviewerID', 'gender', 'age'],
							sep=':+', 
							engine='python')


#create product data
product_data = pd.DataFrame(df.groupby('movieID')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)
# print "product data", len(product_data);
# print product_data.describe();

split_time = df['unixReviewTime'].quantile([.75])[0.75]

after = df[df.unixReviewTime>split_time]
before = df[df.unixReviewTime<=split_time]

user_before = before.reviewerID.unique().tolist()
user_after = after.reviewerID.unique().tolist()

# latent_users = list(set(user_after)-set(user_before))

after_user_data = pd.DataFrame(after.groupby('reviewerID')['overall'].agg(['count']))


#create user data
user_data = pd.DataFrame(before.groupby('reviewerID')['overall'].agg([np.mean, np.std, 'count'])).fillna(1)
# print "user_data"
# print user_data.describe();
latent_users_df = user_data[user_data['count'] <= 10]
latent_users = latent_users_df.index.tolist()
user_data = user_data[user_data['count'] > 10]
print "latent data"
print latent_users_df.describe()

#convert before dataframe to numpy array
numpy_array = before.as_matrix(['reviewerID', 'movieID', 'overall'])

#create product purchase matrix
rows, row_pos = np.unique(numpy_array[:, 0], return_inverse=True)
cols, col_pos = np.unique(numpy_array[:, 1], return_inverse=True)

pivot_table = np.zeros((len(rows), len(cols)), dtype=numpy_array.dtype)
pivot_table[row_pos, col_pos] = numpy_array[:, 2]

result_precision = np.zeros((number_of_thresholds), dtype=np.float64);
result_recall = np.zeros((number_of_thresholds), dtype=np.float64);
result_f_score = np.zeros((number_of_thresholds), dtype=np.float64);

for target in range(len(latent_users)):
	print "target :", target
	# print user_df[user_df.reviewerID==latent_users[target]]
	sim = np.zeros((len(user_before)), dtype=np.float64)	
	for x in range(len(user_before)):
		sim[x] = calcuate_latent_similarity(user_df[user_df.reviewerID==latent_users[target]], user_df[user_df.reviewerID==user_before[x]])
	sim_users = np.argpartition(sim, -number_of_sim_users)[-number_of_sim_users:]
	#print sim_users

	recommend_prob = np.zeros((len(cols)), dtype=np.float64)
	latent_normalize_factor = 0
	for u in sim_users:
		last_purchase_time = before[before.reviewerID==user_before[u]].max(axis=0).iloc[3]
		for x in np.nonzero(pivot_table[u])[0]:
			purchase_time = before[(before.reviewerID==user_before[u]) & (before.movieID==cols[x])].iloc[0, 3]
			latent_factor = 1
			if(purchase_time!=last_purchase_time):
				latent_factor = math.exp(-1.0*seconds_in_a_month/(purchase_time-first_purchase_time))
			recommend_prob[x] += latent_factor
			latent_normalize_factor += latent_factor

	recommend_prob = np.true_divide(recommend_prob, latent_normalize_factor/100.0)
	print recommend_prob.min(), recommend_prob.max()
	for t in range(number_of_thresholds):
		threshold = getThreshold(t)
		recommendation_list = np.argpartition(recommend_prob, -threshold)[-threshold:]
		count=0;
		after_purchased_count = len(after[after.reviewerID==latent_users[target]])
		for x in recommendation_list:
			if(len(after[(after.reviewerID==latent_users[target]) & (after.movieID==cols[x])])>0):
				count+=1
		if len(recommendation_list)>0 :
			precision = count*1.0/len(recommendation_list)
			recall = count*1.0/after_purchased_count
			result_precision[t] += precision
			result_recall[t] += recall
			if((precision+recall)>0):
				f = 2.0*precision*recall/(precision+recall)
				result_f_score[t] += f


	if target>=50:
		break;

result_precision = np.true_divide(result_precision, 51)
result_recall = np.true_divide(result_recall, 51)
result_f_score = np.true_divide(result_recall, 51)

f = open('latent_wrt_list_length.csv', 'w+')
for i in range(number_of_thresholds):
	s = str(getThreshold(i))+", "+str(result_precision[i])+", "+str(result_recall[i])+", "+str(result_f_score[i])+"\n"
	f.write(s)
f.close()

plt.plot([getThreshold(i) for i in range(number_of_thresholds)], result_precision)
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.show()