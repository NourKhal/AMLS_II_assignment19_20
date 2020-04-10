import argparse
import pandas as pd
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation


# Pre-processing the tweets before applying any sentiment classification

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def process_tweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["Tweet"]),tweet["Sentiment"]))
        return processedTweets

    def _processTweet(self, tweet):
        # change all text to lowercase
        tweet = tweet.lower()
        # remove all the meaningless repeated characters from the tweets to unify the words
        tweet = word_tokenize(tweet)
        # remove all links and URLs from the tweets
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        # remove all the hashtag symbols form the tweets
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        # remove all the usernames if any
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
        return [word for word in tweet if word not in self._stopwords]


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='ML model trained on Twitter tweets to classify sentiment from input tweets or messages.')
    arg_parser.add_argument('-i', '--tweets-dir',
                            help='the path to the directory where the tweets and labels are',
                            required=True)


    args = vars(arg_parser.parse_args())
    tweets_directory = args['tweets_dir']
    print("Building sentiment classification model from Twitter tweets (text messages) in {},"
          "Index of sentiment field in the txt file is".format(tweets_directory))

    cwd = os.getcwd()
    csv_file_path = cwd + '/tweets_and_sentiment.csv'
    tweets = pd.read_csv(tweets_directory, sep='\t')
    tweets.columns = ['TweetID', 'Sentiment', 'Tweet']
    dict_tweets = tweets.to_dict('records')
    tweetProcessor = PreProcessTweets()
    preprocessedTrainingSet = tweetProcessor.process_tweets(dict_tweets)






