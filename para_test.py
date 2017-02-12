import pandas as pd 
import numpy as np
import json
import math

number_of_iterations = 11
number_of_thresholds = 10
number_of_products = 200
number_of_sim_users = 5
w_lambda = 0.5
p = 0.5
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

ranges = [x for x in range(1910, 2001, 10) ]

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

def calcuate_similarity(pivot_table, user_data, product_data, i, j, w_lambda):
	if i==j:
		return 0
	normalize_freq = np.max(product_data['count'].values)
	common = (pivot_table[i]*pivot_table[j]).nonzero()

	rating_i = pivot_table[i][common[0]]
	rating_j = pivot_table[j][common[0]]
	rating_i = rating_i - user_data.iloc[i, 0]
	rating_j = rating_j - user_data.iloc[j, 0]
	variance = rating_i*rating_j

	reputation = product_data.iloc[common[0], 0].as_matrix()/5
	frequency = product_data.iloc[common[0], 2]/normalize_freq
	val = np.sum(np.sqrt(w_lambda*np.square(np.reciprocal(reputation))+(1-w_lambda)*np.square(np.reciprocal(frequency)))*variance)
	return val/ ( max(user_data.iloc[i, 1], 1)*max(user_data.iloc[j, 1], 1) )

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
movieDF['year'] = movieDF['name'].apply(lambda x: (int(x[-5:-1])/10)*10)

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
print "Created user data"

#creating genre vector
user_genre_vector = pd.DataFrame(before.groupby('reviewerID')[genre_list].sum()).fillna(1)
user_genre_vector = user_genre_vector.apply(normalizeVector, axis=1)
user_genre_vector = user_genre_vector.rename(columns=genre_reverse)

#creating year vector
user_year_vector = before.pivot_table(index='reviewerID', values=[], columns='year', aggfunc=len, fill_value=0)
year_columns = user_year_vector.columns.values
year_columns_dict = {i:year_columns[i] for i in range(len(year_columns))}
user_year_vector = user_year_vector.apply(normalizeVector, axis=1)
user_year_vector = user_year_vector.rename(columns=year_columns_dict)


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

result_precision = np.zeros((number_of_iterations, number_of_thresholds), dtype=np.float64);
result_recall = np.zeros((number_of_iterations, number_of_thresholds), dtype=np.float64);
result_f_score = np.zeros((number_of_iterations, number_of_thresholds), dtype=np.float64);

for iteration in range(0, number_of_iterations):
	w_lambda = iteration*0.1
	print "\niteration :", iteration
	for target in range(len(accepted_users)):
		if target%10==0:
			print target
		else:
			print target,
		#finding similar users based on genre
		target_genre_vector = user_genre_vector.loc[accepted_users[target]].as_matrix()
		sim = np.array([np.dot(target_genre_vector, user_genre_vector.loc[accepted_users[x]].as_matrix()) for x in range(len(accepted_users))])	
		# genre_sim_users = np.argpartition(sim, -len(sim)/2)[-len(sim)/2:]
		genre_sim_users = np.argpartition(sim, -30)[-30:]

		#finding similar users based on year
		target_year_vector = user_year_vector.loc[accepted_users[target]].as_matrix()
		sim = np.array([np.dot(target_year_vector, user_year_vector.loc[accepted_users[x]].as_matrix()) for x in genre_sim_users])
		# sim_users = np.argpartition(sim, -len(sim)/2)[-len(sim)/2:]
		sim_users = np.argpartition(sim, -20)[-20:]
		year_sim_users = genre_sim_users[sim_users]

		#finding similar users based on reviwe frequency
		target_frequency = user_data.loc[accepted_users[target]].iloc[2]
		sim = np.array([abs(target_frequency-user_data.loc[accepted_users[x]].iloc[2]) for x in year_sim_users])
		# sim_users = np.argpartition(sim, -len(sim)/2)[-len(sim)/2:]
		sim_users = np.argpartition(sim, -10)[-10:]
		frequency_sim_users = year_sim_users[sim_users]

		sim = np.array([calcuate_similarity(pivot_table, user_data, product_data, target, x, w_lambda) for x in frequency_sim_users])	
		sim_users = np.argpartition(sim, -number_of_sim_users)[-number_of_sim_users:]
		sim_users = frequency_sim_users[sim_users]
		sim_users = np.append(sim_users, [target])

		purchase_count = {}
		for u in sim_users:
			for x in np.nonzero(pivot_table[u])[0]:
				if(x in purchase_count):
					purchase_count[x]+=1
				else:
					purchase_count[x]=1

		transition_matrix=np.zeros((len(purchase_count),len(purchase_count)),dtype=np.float64 )
		corelation_matrix=np.zeros((len(purchase_count),len(purchase_count)),dtype=np.float64 )
		purchase_record = purchase_count.keys()

		for u in sim_users:
			user_purchase = pd.DataFrame(before[before['reviewerID']==rows[u]].sort_values('unixReviewTime'))
			purchase_transition = user_purchase['productID'].tolist()
			for i in range(len(purchase_transition)-1):
				x = np.where(cols == purchase_transition[i])[0][0]
				y = np.where(cols == purchase_transition[i+1])[0][0]
				transition_matrix[purchase_record.index(x)][purchase_record.index(y)]+=1

		for i in range(len(transition_matrix)):
			transition_matrix[i] = transition_matrix[i]/purchase_count[purchase_record[i]]

		for group in before[before['productID'].isin([cols[i] for i in purchase_record])].groupby('reviewerID'):
			for x, row_x in group[1].sort_values('unixReviewTime').tail(3).iterrows():
				for y, row_y in group[1].sort_values('unixReviewTime').tail(3).iterrows():
					if x!=y:
						a = np.where(cols == row_x['productID'])[0][0]
						b = np.where(cols == row_y['productID'])[0][0]
						corelation_matrix[purchase_record.index(a)][purchase_record.index(b)]+=1
						corelation_matrix[purchase_record.index(b)][purchase_record.index(a)]+=1

		corelation_matrix = np.reciprocal(np.add(1, np.exp(np.negative(corelation_matrix))))

		transfer_matrix = p*transition_matrix + (1-p)*corelation_matrix;
		last_three_purchase = before[before.reviewerID==rows[target]].tail(3)
		recent_products = [purchase_record.index(np.where(cols == x)[0][0]) for x in last_three_purchase['productID'].tolist()]

		recommend_prob = np.zeros((len(purchase_record)), dtype=np.float64)

		for x in recent_products:
			recommend_prob = recommend_prob + transfer_matrix[x]

		recommend_prob = np.true_divide(recommend_prob, len(recent_products))

		#include trust
		for x in range(len(purchase_record)):
			recommend_prob[x] = alpha*recommend_prob[x] + (1 - alpha)*product_data.iloc[purchase_record[x], 3]
		# print recommend_prob.mean(), recommend_prob.min(), recommend_prob.max()
		for t in range(number_of_thresholds):
			threshold = getThreshold(t)
			recommendation_list = np.argpartition(recommend_prob, -threshold)[-threshold:]
			count=0;
			after_purchased_count = len(after[after.reviewerID==rows[target]])
			for x in recommendation_list:
				if(len(after[(after.reviewerID==rows[target]) & (after.productID==cols[purchase_record[x]])])>0):
					count+=1
			if len(recommendation_list)>0 :
				precision = count*1.0/len(recommendation_list)
				result_precision[iteration][t] += precision
				recall = count*1.0/after_purchased_count
				result_recall[iteration][t] += recall
		if target>=50:
			break;

	np.copyto(result_precision[iteration], np.true_divide(result_precision[iteration], 51))
	np.copyto(result_recall[iteration], np.true_divide(result_recall[iteration], 51))
	np.copyto(result_f_score[iteration], 
		np.divide(2*result_precision[iteration]*result_recall[iteration], 
			result_precision[iteration]+result_recall[iteration]))

f = open('vary_lambda_demo.csv', 'w+')
separator = ", "

f.write("Recommendation list length, "+separator.join(map(str, [i for i in np.arange(0, 1.1, 0.1)]))+"\n")

for i in range(number_of_thresholds):
	s = str(getThreshold(i))+", "+separator.join(map(str, result_precision[:,i]))+"\n"
	f.write(s)
f.write("\n")

f.write("Recommendation list length, "+separator.join(map(str, [i for i in np.arange(0, 1.1, 0.1)]))+"\n")
for i in range(number_of_thresholds):
	s = str(getThreshold(i))+", "+separator.join(map(str, result_recall[:,i]))+"\n"
	f.write(s)
f.write("\n")

f.write("Recommendation list length, "+separator.join(map(str, [i for i in np.arange(0, 1.1, 0.1)]))+"\n")
for i in range(number_of_thresholds):
	s = str(getThreshold(i))+", "+separator.join(map(str, result_f_score[:,i]))+"\n"
	f.write(s)
f.write("\n")

f.close()