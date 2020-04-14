import argparse
import pandas as pd
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


def get_features_and_labels(tweets_directory):
    tweets = pd.read_csv(tweets_directory, sep='\t')
    tweets.columns = ['TweetID', 'Sentiment', 'Tweet']
    tweets_training = tweets[:4199]
    tweets_validation = tweets[4200:5099]
    tweets_test = tweets[5100:6000]
    dict_tweets_training = tweets_training.to_dict('records')
    dict_tweets_validation = tweets_validation.to_dict('records')
    dict_tweets_test = tweets_test.to_dict('records')
    return dict_tweets_training, dict_tweets_validation, dict_tweets_test


# Pre-processing the tweets before applying any sentiment classification

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def process_tweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["Tweet"]),tweet["Sentiment"]))
            random.shuffle(processedTweets)
        return processedTweets

    def _processTweet(self, tweet):
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

    for (words, sentiment) in preprocessed_training_data:
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

    arg_parser = argparse.ArgumentParser(description='ML model trained on Twitter tweets to classify sentiment from input tweets or messages.')
    arg_parser.add_argument('-i', '--tweets-dir',
                            help='the path to the directory where the tweets and labels are',
                            required=True)


    args = vars(arg_parser.parse_args())
    tweets_directory = args['tweets_dir']
    print("Building sentiment classification model from Twitter tweets (text messages) in {}.".format(tweets_directory))

    cwd = os.getcwd()
    csv_file_path = cwd + '/tweets_and_sentiment.csv'
    tweets = pd.read_csv(tweets_directory, sep='\t')
    tweets.columns = ['TweetID', 'Sentiment', 'Tweet']
    dict_tweets = tweets.to_dict('records')
    tweetProcessor = PreProcessTweets()
    preprocessedTrainingSet = tweetProcessor.process_tweets(dict_tweets)

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_features)
    print("MultinomialNB accuracy percent:", classify.accuracy(MNB_classifier, validation_features))

    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(training_features)
    print("BernoulliNB accuracy percent:", classify.accuracy(BNB_classifier, validation_features))

    print("Original Naive Bayes Algo accuracy percent:", (classify.accuracy(classifier, validation_features))*100)
    classifier.show_most_informative_features(15)

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_features)
    print("MNB_classifier accuracy percent:", (classify.accuracy(MNB_classifier, validation_features))*100)

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_features)
    print("BernoulliNB_classifier accuracy percent:", (classify.accuracy(BernoulliNB_classifier, validation_features))*100)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_features)
    print("LogisticRegression_classifier accuracy percent:", (classify.accuracy(LogisticRegression_classifier, validation_features))*100)

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_features)
    print("SGDClassifier_classifier accuracy percent:", (classify.accuracy(SGDClassifier_classifier, validation_features))*100)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_features)
    print("SVC_classifier accuracy percent:", (classify.accuracy(SVC_classifier, validation_features))*100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_features)
    print("LinearSVC_classifier accuracy percent:", (classify.accuracy(LinearSVC_classifier, validation_features))*100)

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_features)
    print("NuSVC_classifier accuracy percent:", (classify.accuracy(NuSVC_classifier, validation_features))*100)



