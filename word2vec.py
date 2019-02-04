# -*- coding: utf8 -*-

#Course : Natural Language Processing, Spring 2018 taught by Prof. Chitta Baral

import numpy as np
import pandas as pd
import re, json, nltk
from nltk.corpus import stopwords
# a tweet tokenizer from nltk.
#from keras.preprocessing.text import Tokenizer 
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
from sklearn.model_selection import train_test_split
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.deprecated.doc2vec import LabeledSentence
LabeledSentence = gensim.models.deprecated.doc2vec.LabeledSentence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale

array = ["morphine", "methadone", "buprenorphine", "hydrocodone", "oxycodone","heroin", "oxycontin", "perc", "percocet","palladone" , "vicodin", "percodan", "tylox" ,"demerol", "oxy", "roxies","opiates", "oxy", "percocet", "percocets", "hydrocodone", "norco",
	    	"norcos", "roxy", "roxies", "roxycodone", "roxicodone", "opana", "opanas", "prozac", "painrelief", "painreliever", "painkillers", "addiction", "opium"]

#making a Dictionary of the Stopwards
words = {}
for i in stopwords.words("english"):
	k = i.encode("utf8")
	words[k] = 0

# regular expressions used to clean up the tweet data
drug = re.compile('|'.join(array).lower())
http_re = re.compile(r'\s+http://[^\s]*')
remove_ellipsis_re = re.compile(r'\.\.\.')
at_sign_re = re.compile(r'\@\S+')
punct_re = re.compile(r"[\"'\[\],.:;()\-&!]")
price_re = re.compile(r"\d+\.\d\d")
number_re = re.compile(r"\d+")

def normalize_tweet(tweet):
    #Regular expressions to replace patterns in the Data
    t = tweet.lower()
    t = re.sub(price_re, 'PRICE', t)
    t = re.sub(remove_ellipsis_re, '', t)
    t = re.sub(drug, 'druginstance', t)
    t = re.sub(http_re, ' LINK', t)
    t = re.sub(punct_re, '', t)
    t = re.sub(at_sign_re, '@', t)
    t = re.sub(number_re, 'NUM', t)
    return t

def feature_extractor(tweet):
	#remove new lines
	#Fixing the upper bound for number of Features. This can be changed.
	max_features = 2000	

	new_tweet = tweet.strip().lower().encode('ascii', errors='ignore')
	new_tweet = normalize_tweet(new_tweet)
	#print new_tweet
	#remove new lines (\n)
	#new_tweet = re.sub(r"\n", " ", new_tweet)
	words_in_tweet = new_tweet.split(" ")
	#Removing stop words from the tweets
	words_in_tweet = [x for x in words_in_tweet if x not in words]
	# for i,j in enumerate(words_in_tweet):
	# 	if j in array:
	# 		words_in_tweet[i] = "druginstance"
	#Decoding them back into ASCII to convert them back into unicode
	new_tweet = " ".join(words_in_tweet).decode("ascii", errors = "ignore")
	#print new_tweet
	#print ("\n")
	
	tokens = tokenizer.tokenize(new_tweet)
	
	#raw_input("please press enter ...")

	return tokens




#open the JSON to a list of dictionaries
with open("Merged_Labelled.json", "r") as f:
	original_data = json.load(f)
data = original_data

#print data[0]
#Turning it into a DataFrame
# cut_off = int (len(data) * 0.90)

# train_data = data[:cut_off]
# test_data = data[cut_off:]

#Creating a list of tuples for the input JSON
formatted_data = [(d["label"],feature_extractor(d["tweet"])) for d in data if d["label"] != "None"]
#test_set = [(d["label"],feature_extractor(d["tweet"])) for d in test_data]
#Creating DataFrame and removing data with Sentiments as None and other invalid sentiment values. 
#This is done because the LSTM expects the Class label to be a Floating point number (Probabilities)
df = pd.DataFrame(formatted_data, columns = ["sentiment", "tweet"])
df = df[df["sentiment"]!= "None"]
df = df[df["sentiment"]!= "none"]
df = df[df["sentiment"] != "10"]
df = df[df["sentiment"] != "01"]
#print df.head(10)["tweet"]
#removing the itemid
#df.drop("index", inplace = True, axis = 1)
#Creating the TEST TRAIN Split using sklearn, with 30% of data as Test.
x_train, x_test, y_train, y_test = train_test_split(np.array(df.tweet),
                                                    np.array(df.sentiment), test_size=0.3)
#Converting the Sentiment labels to Float
y_train = [map(float, x) for x in y_train]
y_test = [map(float, x) for x in y_test]

#print y_train


#Converting the Tokens into Labelled Sentence object. 
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train_label = labelizeTweets(x_train, 'TRAIN')
x_test_label = labelizeTweets(x_test, 'TEST')
#Output of the format:
#
#print x_train[3][0]

#we set the dimensions to 200 (Default). sliding window 10. building Word2vec for words that I have occured a minimum times of 5. 
word_to_vec = Word2Vec(size = 200, window = 10, min_count=10, workers = 11, alpha = 0.025, iter = 20)
word_to_vec.build_vocab([x[0] for x in x_train_label])
#Number of words for which vectors are built
m = word_to_vec.corpus_count
#print m
#Training. 20 iterations
word_to_vec.train([x[0] for x in x_train], epochs = word_to_vec.iter, total_examples = m)


#print tweet_w2v["druginstance"]

#final_embedding = tweet_w2v._nemb_final.eva
#The weight vectors generated from the word2vec are stored in the syn0 dictionary
pretrained_weights = word_to_vec.wv.syn0
vocabulary_size, size_embedding = pretrained_weights.shape
#print pretrained_weights.shape



#Trying to Buid the LSTM network
print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
#fitting the TFIDF for the X-train-label
matrix = vectorizer.fit_transform([x[0] for x in x_train_label])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
#print 'vocab : ', tfidf

#print tfidf[u"druginstance"]
#function that given a list of tokens, creates an averaged TWeet vector
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += word_to_vec[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. 
            continue
    if count != 0:
        vec /= count
    return vec
#Converted X_train and X_test into list of vectors and also scaling them so that they have a Mean of 0
#and a Standard Deviation of 1
train_word_to_vec = np.concatenate([buildWordVector(z, 200) for z in map(lambda x: x[0], x_train_label)])
train_vecs_w2v = scale(train_word_to_vec)

test_word_to_vec = np.concatenate([buildWordVector(z, 200) for z in map(lambda x: x[0], x_test_label)])
test_word_to_vec = scale(test_word_to_vec)

#we are using a 2 layer LSTM. 100 neurons in Dense layer 1 and layer 2 has 1.

model = Sequential()
#USing the Keras Built-in Optimizer. Relu for 1st layer and Sigmoid for the next.
model.add(Dense(100, activation='relu', input_dim=200))
#model.add(Dense(100, activation = "relu", input_dim = 200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#Set the number of epochs to be 75, batch_size to 32 and verbose to 2 to see as much info of training process.
model.fit(train_word_to_vec, np.array(y_train), epochs=75, batch_size=32, verbose=2)

#Highest Accuracy achieved 57.35%
score = model.evaluate(test_word_to_vec, np.array(y_test), batch_size=8, verbose=2)
print "the accuracy is", score[1]
