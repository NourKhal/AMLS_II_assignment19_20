import argparse
import pandas as pd
import os




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






