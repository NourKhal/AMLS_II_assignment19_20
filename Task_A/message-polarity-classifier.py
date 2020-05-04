import argparse
import collections
import pickle
import random
import re
from pprint import PrettyPrinter
from string import punctuation

import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import stopwords
from nltk.metrics.scores import (precision, recall)
from nltk.tokenize import word_tokenize


# Load the tweets from the tsv as a pandas dataframe
def load_tweets(tweet_file_path):
    tweets_dataframe = pd.read_csv(tweet_file_path, sep='\t', header=None, quotechar="'")
    tweets_dataframe.columns = ['TweetID','Sentiment', 'Tweet']
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


# Remove all the unavailable tweets from the dataframe
def remove_unavailable_tweets(tweets_dataframe):
    print("Filtering out unavailable tweets...")
    tweets_dataframe = tweets_dataframe[tweets_dataframe['Tweet'] != 'Not Available']
    print("Filtered down to {} tweets".format(len(tweets_dataframe.index)))
    return tweets_dataframe


# Pre-processing the tweets before applying any sentiment classification
class TweetPreprocessor:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def preprocess(self, list_of_tweets):
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
        # tokenise the tweet into words
        tweet = word_tokenize(tweet)
        return [word for word in tweet if word not in self._stopwords]


# Function to build the vocabulary of all the words that exist in the tweets
def create_wordbook(preprocessed_training_data):
    all_words = []

    for (words,sentiment) in preprocessed_training_data:
        all_words.extend(words)

    word_list = FreqDist(all_words) # encode the frequency distributions and count the number of each occurrence
    word_features = word_list.keys()
    return word_features


def extract_tweet_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


def construct_featureset(tweets_dict, preprocessor):
    preprocessed_tweets = preprocessor.preprocess(tweets_dict)
    return classify.apply_features(extract_tweet_features, preprocessed_tweets)


def build_model(training_features,preprocessed_validation_data ):
    NBClassifier = NaiveBayesClassifier.train(training_features)
    predictions = [NBClassifier.classify(extract_tweet_features(tweet[0])) for tweet in preprocessed_validation_data]
    return NBClassifier, predictions


def evaluate_model(NBClassifier):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    accuracy = classify.accuracy(NBClassifier, validation_features) * 100

    for i, (feats, label) in enumerate(validation_features):
        refsets[label].add(i)
        observed = NBClassifier.classify(feats)
        testsets[observed].add(i)
        negative_precision = precision(refsets['negative'], testsets['negative'])
        neutral_precision = precision(refsets['neutral'], testsets['neutral'])
        positive_precision = precision(refsets['positive'], testsets['positive'])
        positive_recall = recall(refsets['positive'], testsets['positive'])
        neutral_recall = recall(refsets['neutral'], testsets['neutral'])
        negative_recall = recall(refsets['negative'], testsets['negative'])
        try:
            avg_recall = (1/3)*(negative_recall + positive_recall + neutral_recall)
            avg_precision = (1/3)*(negative_precision + positive_precision + neutral_precision)
            print(accuracy, avg_recall, avg_precision)
        except TypeError:
            pass

def save_model(MaxEntClassifier):
    f = open('NBClassifier.pickle', 'wb')
    pickle.dump(MaxEntClassifier, f)
    f.close()
    return trained_model

def restore_trained_model(trained_model):
    f = open(trained_model, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        description='ML model trained on Twitter tweets to classify sentiment from input tweets or messages.')
    arg_parser.add_argument('-i', '--tweets-file',
                            help='the path to the directory where the tweets and labels are',
                            required=True)

    args = vars(arg_parser.parse_args())
    tweets_file = args['tweets_file']
    print("Building sentiment classification model from Twitter tweets (text messages) in {}.".format(tweets_file))

    tweets_df = load_tweets(tweets_file)
    tweets_df = tweets_df.drop_duplicates()
    validate_against_file(tweets_file, tweets_df)
    tweets_df = remove_unavailable_tweets(tweets_df)

    training_df, validation_df, test_df = np.split(tweets_df.sample(frac=1), [int(.6*len(tweets_df)), int(.8*len(tweets_df))])
    preprocessor = TweetPreprocessor()
    word_features = create_wordbook(preprocessor.preprocess(training_df.to_dict('records')))
    training_features = construct_featureset(training_df.to_dict('records'), preprocessor)
    validation_features = construct_featureset(validation_df.to_dict('records'), preprocessor)
    test_features = construct_featureset(test_df.to_dict('records'), preprocessor)
    preprocessed_validation_data = preprocessor.preprocess(validation_df.to_dict('records'))
    NBClassifier, predictions = build_model(training_features, preprocessed_validation_data)
    evaluate_model(NBClassifier)

    trained_model = save_model(NBClassifier)
    classifier = restore_trained_model(trained_model)
    test_accuracy = classify.accuracy(classifier, test_features)*100
