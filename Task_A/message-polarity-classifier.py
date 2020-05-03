import argparse
import collections
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
        # tokenise the tweet into words
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

    arg_parser = argparse.ArgumentParser(
        description='ML model trained on Twitter tweets to classify sentiment from input tweets or messages.')
    arg_parser.add_argument('-i', '--tweets-dir',
                            help='the path to the directory where the tweets and labels are',
                            required=True)

    args = vars(arg_parser.parse_args())
    tweets_directory = args['tweets_dir']
    print("Building sentiment classification model from Twitter tweets (text messages) in {}.".format(tweets_directory))

    dict_tweets_training, dict_tweets_validation, dict_tweets_test = get_features_and_labels(tweets_directory)

    tweetProcessor = PreProcessTweets()
    preprocessed_training_data = tweetProcessor.process_tweets(dict_tweets_training)
    preprocessed_validation_data = tweetProcessor.process_tweets(dict_tweets_validation)
    preprocessed_test_data = tweetProcessor.process_tweets(dict_tweets_test)

    word_features = create_wordbook(preprocessed_training_data)
    training_features = classify.apply_features(extract_tweet_features, preprocessed_training_data)
    validation_features = classify.apply_features(extract_tweet_features, preprocessed_validation_data)
    test_features = classify.apply_features(extract_tweet_features, preprocessed_test_data)

    NBayesClassifier = NaiveBayesClassifier.train(training_features)
    print("Classifier accuracy percent:",classify.accuracy(NBayesClassifier, test_features)*100)
    NBResultLabels = [NBayesClassifier.classify(extract_tweet_features(tweet[0])) for tweet in preprocessed_validation_data]
    most_informative_features = NBayesClassifier.show_most_informative_features(15)

    # # get the majority vote
    if NBResultLabels.count('positive') > NBResultLabels.count('negative') and NBResultLabels.count('neutral'):
        print("Overall Positive Sentiment")
        print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
    elif NBResultLabels.count('negative') > NBResultLabels.count('positive') and NBResultLabels.count('neutral'):
        print("Overall neutral Sentiment")
        print("Neutral Sentiment Percentage = " + str(100*NBResultLabels.count('neutral')/len(NBResultLabels)) + "%")
    else:
        print("Overall Negative Sentiment")
        print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
    save_classifier = open("naivebayes.pickle","wb")
    pickle.dump(NBayesClassifier, save_classifier)
    save_classifier.close()

    classifier_f = open("naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()

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

    SVC_classifier = SklearnClassifier(SVC(kernel='poly', gamma=0.0001, C=100))
    SVC_classifier.train(training_features)
    print("SVC_classifier accuracy percent:", (classify.accuracy(SVC_classifier, validation_features))*100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_features)
    print("LinearSVC_classifier accuracy percent:", (classify.accuracy(LinearSVC_classifier, validation_features))*100)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(test_features):
             refsets[label].add(i)
             observed = NBayesClassifier.classify(feats)
             testsets[observed].add(i)
    print('negative precision:', precision(refsets['negative'], testsets['negative']))
    print('neutral precision:', precision(refsets['neutral'], testsets['neutral']))
    print('positive precision:', precision(refsets['positive'], testsets['positive']))

    print('negative recall:', recall(refsets['negative'], testsets['negative']))
    print('neutral recall:', recall(refsets['neutral'], testsets['neutral']))
    print('positive recall:', recall(refsets['positive'], testsets['positive']))


