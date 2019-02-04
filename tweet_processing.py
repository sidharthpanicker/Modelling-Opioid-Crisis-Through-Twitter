#!/usr/bin/python 2.7
# -*- coding: utf8 -*-

#Course : Natural Language Processing, Spring 2018 taught by Prof. Chitta Baral


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
import nltk.classify
import nltk
import json, string, re

#Drug keywords that indicate Opioid use
array = ["morphine", "methadone", "buprenorphine", "hydrocodone", "oxycodone","heroin", "oxycontin", "perc", "percocet","palladone" , "vicodin", "percodan", "tylox" ,"demerol", "oxy", "roxies","opiates", "oxy", "percocet", "percocets", "hydrocodone", "norco",
	    	"norcos", "roxy", "roxies", "roxycodone", "roxicodone", "opana", "opanas", "prozac", "painrelief", "painreliever", "painkillers", "addiction", "opium"]

words = {}
for i in stopwords.words("english"):
	k = i.encode("utf8")
	words[k] = 0

# regular expressions used to clean up the tweet data
mrt_station_re = re.compile('|'.join(array).lower())
http_re = re.compile(r'\s+http://[^\s]*')
remove_ellipsis_re = re.compile(r'\.\.\.')
at_sign_re = re.compile(r'\@\S+')
punct_re = re.compile(r"[\"'\[\],.:;()\-&!]")
price_re = re.compile(r"\d+\.\d\d")
number_re = re.compile(r"\d+")

def normalize_tweet(tweet):
    t = tweet.lower()
    t = re.sub(price_re, 'PRICE', t)
    t = re.sub(remove_ellipsis_re, '', t)
    t = re.sub(mrt_station_re, 'druginstance', t)
    t = re.sub(http_re, ' LINK', t)
    t = re.sub(punct_re, '', t)
    t = re.sub(at_sign_re, '@', t)
    t = re.sub(number_re, 'NUM', t)
    return t

def feature_extractor(tweet):
	#remove new lines
	features_per_tweet = {}
	new_tweet = tweet.strip().lower().encode('ascii', errors='ignore')
	new_tweet = normalize_tweet(new_tweet)
	print new_tweet
	#remove new lines (\n)
	#new_tweet = re.sub(r"\n", " ", new_tweet)
	words_in_tweet = new_tweet.split(" ")
	words_in_tweet = [x for x in words_in_tweet if x not in words]
	# for i,j in enumerate(words_in_tweet):
	# 	if j in array:
	# 		words_in_tweet[i] = "druginstance"
	new_tweet = " ".join(words_in_tweet).decode("ascii", errors = "ignore")
	#print new_tweet
	#print ("\n")

	for trigrams in nltk.trigrams(new_tweet.split(" ")):
		features_per_tweet["presence(%s,%s,%s)" % (trigrams[0],trigrams[1],trigrams[2])] = True

	print features_per_tweet
	#raw_input("please press enter ...")

	return features_per_tweet



#open the JSON to a list of dictionaries
with open("Merged_Labelled.json", "r") as f:
	original_data = json.load(f)
data = original_data

#feature_extractor(data[0]["tweet"])
#assuming a test split of 0.1
cut_off = int (len(data) * 0.90)

train_data = data[:cut_off]
test_data = data[cut_off:]

train_set = [(feature_extractor(d["tweet"]),d["label"]) for d in train_data]
test_set = [(feature_extractor(d["tweet"]),d["label"]) for d in test_data]


for i in xrange(4):
	print train_set[i][0]
	print ("\n")
	print train_set[i][1]
	print ("\n")
	print ("\n")
	print ("\n")

# classifier = nltk.NaiveBayesClassifier

# classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier = nltk.classify.SklearnClassifier(LinearSVC())
classifier.train(train_set)

#classifier.show_most_informative_features(20)

# collect tweets that were wrongly classified
errors = []
for d in test_set:
    label = d[1]
    guess = classifier.classify(d[0])
    print guess
    #print "guess", guess, "label", label
    if guess != label:
        errors.append( (label, guess, d) )

# for (label, guess, d) in sorted(errors):
#     print 'correct label: %s\nguessed label: %s\ntweet=%s\n' % (label, guess, d['tweet'])

print 'Total errors: %d' % len(errors)

print 'Accuracy: ', nltk.classify.accuracy(classifier, test_set)








	    	

