import os
import re
from pyspark.sql.functions import *
from pyspark.sql.types import *
from bs4 import BeautifulSoup
import preprocessor as p 
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from pyspark.sql import SparkSession

import logging
import pandas as pd

#nltk.download('stopwords')
#nltk.download('wordnet')
logging.basicConfig(filename = 'Tweets.log',level = logging.DEBUG)
def read_tweets(spark):
    filepath = os.path.join(os.getcwd(),'Q7')
    print(filepath)
    df = spark.read.csv(os.path.join(filepath,os.listdir(filepath)[1]))
    df = df.withColumnRenamed('_c0','Target')
    df = df.withColumnRenamed('_c1','TweetID')
    df = df.withColumnRenamed('_c2','Time/Date')
    df = df.withColumnRenamed('_c3','Flag')
    df = df.withColumnRenamed('_c4','UserID')
    df = df.withColumnRenamed('_c5','Tweet')
    df = df.select('Target','Time/Date','UserID','Tweet')
    logging.info('Read Dataframe')
    return df

def process_tweets(df):
    def get_date(x):
        return ' '.join([x.split(' ')[2],x.split(' ')[1],x.split(' ')[-1]])
    def get_time(x):
        return x.splt(' ')[3]
    def extract_hashtag(x):
        return re.findall(r'#(\w+)',x)
    def process_html(x):
        soup = BeautifulSoup(x,'html.parser')
        return soup.get_text()
    def extract_mentions(x):
        return re.findall(r'@(\w+)',x)
    def clean_tweet(x):
         p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.HASHTAG,p.OPT.MENTION)
         return p.clean(x)
    def remove_stopwords_numbers_punctuation(x):
        tokenizer = TweetTokenizer()
        stop = stopwords.words('english')
        x = re.sub(r'[\d]',"",re.sub(r'[^\w\s]',"",x))
        return [word for word in tokenizer.tokenize(x) if word not in stop]
    def turn_to_lower(words):
        return [x.lower() for x in words]

    returndate = udf(get_date,StringType())
    returntime = udf(get_time,StringType())
    extracthashtag = udf(extract_hashtag,ArrayType(StringType()))
    processhtml = udf(process_html,StringType())
    extractmentions = udf(extract_mentions,ArrayType(StringType()))
    cleantweet = udf(clean_tweet,StringType())
    removestop_punc_num = udf(remove_stopwords_numbers_punctuation,ArrayType(StringType()))
    turntolower = udf(turn_to_lower,ArrayType(StringType()))

    df = df.select('*',returndate(df['Time/Date']).alias('Date'))
    df = df.select('*',returntime(df['Time/Date']).alias('Time'))
    logging.info('Processed Time and Date')
    df = df.select('*',extracthashtag(df["Tweet"]).alias('Hashtag'))
    logging.info('Extracted Hashtag')
    df = df.select('*',processhtml(df["Tweet"]).alias('Text_HTML'))
    logging.info('Converted HTML')
    df = df.select('*',extractmentions(df["Tweet"]).alias('Mentions'))
    logging.info('Extracted Mentions')
    df = df.select('Target','UserID','Tweet','Date','Time','Hashtag','Text_HTML','Mentions')
    df = df.select('Target','UserID','Tweet','Date','Time','Hashtag','Mentions',cleantweet(df['Text_HTML']).alias('Text_Clean'))
    df2 = df.select('Target','UserID','Tweet','Date','Time','Hashtag','Mentions',removestop_punc_num(df['Text_Clean']).alias('Text_Stop'))
    logging.info('Removed Stop Words, Punctuations and Numbers')
    df3 = df2.select('Target','UserID','Tweet','Date','Time','Hashtag','Mentions',turntolower(df2["Text_Stop"]).alias('Text_Lower'))
    logging.info('urned to lower case')
    return df3


if __name__ == "__main__":
    spark = SparkSession.builder.appName("TweetMining").getOrCreate()
    df = read_tweets(spark)
    df = process_tweets(df)
    pandas_df = df.toPandas()
    pandas_df.to_csv('TweetsProcessed.csv')



    