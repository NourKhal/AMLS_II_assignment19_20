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
import collections
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.metrics.scores import (precision, recall)


def get_features_and_labels(tweets_directory):
    tweets = pd.read_csv(tweets_directory, sep='\t')
    tweets.columns = ['TweetID', 'Sentiment', 'Tweet']
    tweets = tweets[tweets['Tweet'] != 'Not Available']
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


