import argparse
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk import FreqDist, MaxentClassifier, classify, precision, recall, collections
from pprint import PrettyPrinter
import random
import numpy as np


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
        # remove all the meaningless repeated characters from the tweets to unify the words
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
    algorithm = MaxentClassifier.ALGORITHMS[0]
    MaxEntClassifier = MaxentClassifier.train(training_features, algorithm,max_iter=3)
    predictions =  [MaxEntClassifier.classify(extract_tweet_features(tweet[0])) for tweet in preprocessed_validation_data]
    return MaxEntClassifier, predictions


def evaluate_model(MaxEntClassifier):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    accuracy = classify.accuracy(MaxEntClassifier, validation_features)*100
    accuracy_list.append(accuracy)

    for i, (feats, label) in enumerate(validation_features):
        refsets[label].add(i)
        observed = MaxEntClassifier.classify(feats)
        testsets[observed].add(i)
        negative_precision = precision(refsets['negative'], testsets['negative'])
        positive_precision = precision(refsets['positive'], testsets['positive'])
        positive_recall = recall(refsets['positive'], testsets['positive'])
        negative_recall = recall(refsets['negative'], testsets['negative'])
        try:
            avg_recall = 0.5*(negative_recall+positive_recall)
            avg_precision = 0.5*(negative_precision+positive_precision)
            precision_list.append(avg_precision)
            recall_list.append(avg_recall)
        except TypeError:
            pass
    Average_recall = np.mean(recall_list)
    Average_precission = np.mean(precision_list)
    Average_accuracy = np.mean(accuracy_list)
    F1_score = 2*((np.mean(recall_list) * np.mean(precision_list)) / (np.mean(recall_list) + np.mean(precision_list)))
    return Average_accuracy, Average_precission, Average_recall, F1_score


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        description='ML model trained on Twitter tweets to classify sentiment from input tweets or messages.')
    arg_parser.add_argument('-i', '--tweets-file',
                            help='The path to the TSV tweets file containing labelled tweets',
                            required=True)

    args = vars(arg_parser.parse_args())
    tweet_file = args['tweets_file']
    print("Building sentiment classification model from Twitter tweets (text messages) in {}.".format(tweet_file))

    tweets_df = load_tweets(tweet_file)
    validate_against_file(tweet_file, tweets_df)
    tweets_df = remove_unavailable_tweets(tweets_df)

    df_grouped = tweets_df.groupby('Topic')
    for name, group in df_grouped:
        training_df, validation_df, test_df = np.split(group.sample(frac=1), [int(.6*len(group)), int(.8*len(group))])
        preprocessor = TweetPreprocessor()
        word_features = create_wordbook(preprocessor.preprocess(training_df.to_dict('records')))
        training_features = construct_featureset(training_df.to_dict('records'), preprocessor)
        validation_features = construct_featureset(validation_df.to_dict('records'), preprocessor)
        test_features = construct_featureset(test_df.to_dict('records'), preprocessor)
        preprocessed_validation_data = preprocessor.preprocess(validation_df.to_dict('records'))
