import argparse
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk import FreqDist
from pprint import PrettyPrinter
import random


# Load the tweets from the tsv as a pandas dataframe
def load_tweets(tweet_file_path):
    tweets_dataframe = pd.read_csv(tweet_file_path, sep='\t', header=None, quotechar="'")
    tweets_dataframe.columns = ['TweetID', 'Topic', 'Sentiment', 'Tweet']
    print("Read in {} tweets".format(len(tweets_dataframe.index)))
    return tweets_dataframe


def validate_against_file(tweet_file_path, tweets_dataframe):
    # make sure all rows in the file have been loaded
    raw_tweet_list = open(tweet_file_path).readlines()
    raw_tweet_ids = [int(line.split('\t')[0]) for line in raw_tweet_list]
    found_list = []
    for index, row in tweets_dataframe.iterrows():
        tweet_id = row['TweetID']
        if tweet_id in raw_tweet_ids:
            found_list.append(tweet_id)
    not_found_list = set(raw_tweet_ids) - set(found_list)
    if not_found_list:
        print("Tweets not parsed into data frame:\n{}".format(list(not_found_list)))
        not_found_rows = [line for line in raw_tweet_list if int(line.split('\t')[0]) in not_found_list]
        PrettyPrinter().pprint(not_found_rows)



# Pre-processing the tweets before applying any sentiment classification
class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def process_tweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self.clean(tweet["Tweet"]), tweet["Sentiment"]))
            random.shuffle(processedTweets)
        return processedTweets

    def clean(self, tweet):
        # change all text to lowercase
        tweet = tweet.lower()
        # remove all links and URLs from the tweets
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        # remove all the usernames if any
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
        # remove all the hashtag symbols form the tweets
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        # remove all the meaningless repeated characters from the tweets to unify the words
        tweet = word_tokenize(tweet)
        return [word for word in tweet if word not in self._stopwords]


# Function to build the vocabulary of all the words that exist in the tweets
def create_wordbook(preprocessed_training_data):
    all_words = []

    for (words,sentiment) in preprocessed_training_data:
        all_words.extend(words)

    word_list = FreqDist(all_words)
    word_features = word_list.keys()
    return word_features


def extract_tweet_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        description='ML model trained on Twitter tweets to classify sentiment from input tweets or messages.')
    arg_parser.add_argument('-i', '--tweets-dir',
                            help='the path to the directory where the tweets and labels are',
                            required=True)

    args = vars(arg_parser.parse_args())
    tweets_directory = args['tweets_dir']
    print("Building sentiment classification model from Twitter tweets (text messages) in {}.".format(tweets_directory))
