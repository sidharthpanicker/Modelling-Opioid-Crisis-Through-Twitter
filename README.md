# Modelling Opioid Crisis through Twitter
COurse: Natural Language Processing

Tools used : Python (Keras, Tensorflow, Gensim, NLTK, Pandas, Numpy, scikit-learn, re, json)

This aim of the project is to Model the Dynamics of the Opioid Crisis through Twitter.

Dealing with the Data: When I found corpus for the drug-related tweets, they had tweets, with mentions about all sorts of drugs to it. But, for the sake of this project I stuck only with tweets which are related to Opioids. I looked for terms that are representative of Opioid usage and came up with the following array and formulated that any tweet which contains these words is Opioid related.

array = ["morphine", "methadone", "buprenorphine", "hydrocodone", "oxycodone" "heroin", "oxycontin", "perc", "percocet","palladone" , "vicodin", "percodan", "tylox" ,"demerol", "oxy", "roxies","opiates", "oxy", "percocet", "percocets", "hydrocodone", "norco", "norcos", "roxy", "roxies", "roxycodone", "roxicodone", "opana", "opanas", "prozac", "painrelief", "painreliever", "painkillers", "addiction", "opium"]

I then labelled them as: “None”: If the Tweet is not at all Drug related. “0”: If the tweet is drug related but not, abuse. “1”: If the tweet is drug related and also abuse.

Data Pre-processing: I performed various Data Processing steps, like the drug keywords are replaced by "Drug Instance" and all the numbers by "Num" and http links are replaced by LINK and all punctuation by "" etc.

Extracting Features: I decided to use the Trigram representatiion of the words and for this I decided to use the Trigrams feature from the nltk library. The output of this feature extraction would be fed to the Classifier (SVM and Naive Bayes)

Classifier: The SVM and NB gave us an approximate accuracy of 62%, state of the art being around 75%. I attribute our low accuracy to the less training size (approximately around 1800 tweets). If I could, train with higher number of tweets, I'm sure it would've done so much better.

Word2Vec and LSTM: As an attempt to raise the accuracy of the model, I decided to build a word2vec representation, using the Skipgram model and pass the weights trained in the first layer (which is stored in syn0 dictionary) as input to a deep LSTM network. This approach gave us an accuracy of 61%, but this is again due to the less amount of Training data, thats required to train a DNN.
