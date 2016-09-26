import pandas as pd 
import numpy as np
import json

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

df = getDF('Digital_Music_5.json')
#df = getDF('test_50.json')
# df = df[['reviewerID', 'asin', 'overall']]
# df.to_csv("test.csv")

#result = df.sort_values(by="reviewerID")
#print result
#print df["reviewerID"].max()

#create user dataframe users

users = df.reviewerID.unique()
print len(users)
users = asin.unique()
print len(products)

# #create product dataframe
#products = df.groupby(['asin']).size()

# #products['count'] = products.transform(size)


# #products = df.groupby("asin").apply(productGroup)
#print products
# #print products.count()
#print resulr