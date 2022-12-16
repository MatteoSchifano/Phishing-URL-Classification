import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import re, string
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

sw = stopwords.words('english')
sw.remove('not')

def clean_tweet(tweet):
    
    tweet = tweet.lower()
    tweet = tweet.replace('\n', ' ')
    tweet = re.sub("'", "", tweet) 
    tweet = re.sub("@[A-Za-z0-9_]+","", tweet)
    tweet = re.sub("#[A-Za-z0-9_]+","", tweet)
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub('[()!?]', ' ', tweet)
    tweet = re.sub('\[.*?\]',' ', tweet)
    tweet = re.sub("[^a-z0-9]"," ", tweet)
    tweet = re.sub(' +', ' ', tweet)
    tweet = tweet.split()
    tweet = [w for w in tweet if not w in sw]
    tweet = " ".join(word for word in tweet)
    return tweet

def make_tweets_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    x, y = preprocessing(df)
    
    return x, y

def preprocessing(df_: pd.DataFrame) -> pd.DataFrame:
    df = df_.copy()
    del df['Unnamed: 0']
    del df['Source of Tweet']
    del df['Date Created']
    del df['Number of Likes']
    
    lb_sent = LabelEncoder()
    df['Sentiment'] = lb_sent.fit_transform(df.Sentiment)
    
    df['clean_tweet'] = df.Tweet.apply(lambda x: clean_tweet(x))

    tfidf = TfidfVectorizer(tokenizer=word_tokenize, min_df=10, max_df=0.90)
    X = tfidf.fit_transform(df.clean_tweet)
    y = df.loc[:, 'Sentiment']
    
    return X, y



def make_sarcasm_dataset(path: str, path2: str) -> tuple[np.ndarray, np.ndarray]:
    df  = pd.read_json(path, lines=True)
    df2 = pd.read_json(path2, lines=True)

    df = pd.concat([df, df2])
    
    df = text_preprocessing(df)
    
    X = df.loc[:, df.columns != 'is_sarcastic'].iloc[:, 2:]
    y = df.loc[:, 'is_sarcastic']
    
    return X, y
    

def text_cleaning(x, lemm: WordNetLemmatizer):
    
    x = x.lower()
    x = re.sub('\s+\n+', ' ', x)
    x = re.sub('[^a-zA-Z0-9]', ' ', x)
    x = x.split()
    
    x = [lemm.lemmatize(word, "v") for word in x if not word in sw]
    x = ' '.join(x)
    
    return x


def text_preprocessing(df_):
    lemm = WordNetLemmatizer()
    df = df_.copy()
    
    df['text_clean'] = df.headline.apply(lambda x: text_cleaning(x, lemm))
    df['sentence_length'] = df.text_clean.apply(lambda x: len(x.split()))
    df = df[['headline', 'text_clean','sentence_length','is_sarcastic']]
    
    cv = CountVectorizer(tokenizer=word_tokenize, min_df=10, max_df=0.60, dtype=np.int32)
    X = cv.fit_transform(df.text_clean)

    df3 = add_sparse_matrix_to_dataframe(X, cv.vocabulary_, df)
    
    return df3


def add_sparse_matrix_to_dataframe(sm, columns: list[str], df_: pd.DataFrame):
    df = df_.copy()
    
    sm_df = pd.DataFrame(sm.todense(), columns=columns)
    
    df.reset_index(inplace=True, drop=True)
    sm_df.reset_index(inplace=True, drop=True)

    return pd.concat([df, sm_df], axis = 1)


if __name__ == '__main__':
    
    X, y = make_tweets_dataset('.\data\\fifa_world_cup_2022_tweets.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    print(accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
    
    exit()
    
    
    X, y = make_sarcasm_dataset('.\data\Sarcasm_Headlines_Dataset.json','.\data\Sarcasm_Headlines_Dataset_v2.json')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    model = RandomForestClassifier(n_estimators=50, max_depth=20)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    print(accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))