# Esse notebook explica o passo-a-passo de como construir uma analise de sentimento usando dados do twitter.
# Python 3
##
##
#Part 1: Collecting data 
#Part 2: Text Pre-processing 
#Part 3: Term Frequencies 
#Part 4: Rugby and Term Co-Occurrences 
#Part 5: Data Visualisation Basics 
#Part 6: Sentiment Analysis Basics (this article) 
##
##
import tweepy    # http://www.tweepy.org/
from tweepy import OAuthHandler
# abaixo omiti as minhas informações. Use as suas.
consumer_key = 'xxxxxx'
consumer_secret = 'xxxxxxx'
access_token = 'xxxxxxxxx'
access_secret = 'xxxxxxxxxxxxxxxxxxxxxx' .  #use suas credenciais
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)
#######
# vamos ver o que esta no meu twitter - apenas 10
for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    print(status.text)
######
def process_or_store(tweet):
    print(json.dumps(tweet))
import json
# save results as json format
for status in tweepy.Cursor(api.home_timeline).items(10):
    # Process a single status
    process_or_store(status._json)
# a list of my friends in twitter
for friend in tweepy.Cursor(api.friends).items():
    process_or_store(friend._json)
# all my twitters
for tweet in tweepy.Cursor(api.user_timeline).items():
    process_or_store(tweet._json)
from tweepy import Stream
from tweepy.streaming import StreamListener
class MyListener(StreamListener):
    def on_data(self, data):
        try:
            with open('python.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True
#####
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#bitcoin'])
#########
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing

# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
consumer_key = 'xxxxxxx'
consumer_secret = 'xxxxxxxxx'
access_token = 'xxxxxxx'
access_secret = 'xxxxxxxx'

# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
####
# Vamos criar um objeto para extrair as informações:
extractor = twitter_setup()
# criando uma lista de twitters:
tweets = extractor.user_timeline(screen_name="realDonaldTrump", count=2000)
print("Number of tweets extracted: {}.\n".format(len(tweets)))
# imprimindo os 5 twitters mais recentes:
print("5 recent tweets:\n")
for tweet in tweets[:5]:
    print(tweet.text)
    print()
#####
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
display(data.head(10))
    # Return API with authentication:
    api = tweepy.API(auth)
    return api
######
print(dir(tweets[0]))
print(tweets[0].retweet_count)
print(tweets[0].geo)
print(tweets[0].coordinates)
print(tweets[0].entities)
data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])
display(data.head(10))
#######
mean = np.mean(data['len'])
print("The lenght's average in tweets: {}".format(mean))
######
# extraindo os twitters com mais FAVs e RTs:
fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])
fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]
# Max FAVs:
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))
# Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))
######
# criando uma visualizacao em time series dos twitters
tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])
# comprimento ao longo do tempo:
tlen.plot(figsize=(16,4), color='r');
# como visualizar Likes vs retweets:
tfav.plot(figsize=(16,4), label="Likes", legend=True)
#####
# analisando a origem dos twitters:
sources = []
for source in data['Source']:
    if source not in sources:
        sources.append(source)

# imprimindo a lista de devices:
print("Creation of content sources:")
for source in sources:
    print("* {}".format(source))
####
# Vamos usar o numpy para analisar os twitters e mapear:
percent = np.zeros(len(sources))
for source in data['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass
percent /= 100

# Pie chart:
pie_chart = pd.Series(percent, index=sources, name='Sources')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));
tret.plot(figsize=(16,4), label="Retweets", legend=True);
####
from textblob import TextBlob
import re

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Essa parte cria uma funcao para classificar a polaridade de um twitter usando textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
#####
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])
display(data.head(10))
#####
# construindo uma lista com os twitters classificados:
pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]
###
# Imprimindo os percentuais:
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))
####
