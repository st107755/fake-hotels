from operator import index
import pdb
from nltk.corpus.reader import rte
from nltk.sem.evaluate import Model
from numpy import random
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    plot_precision_recall_curve,
)
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler,TomekLinks,OneSidedSelection
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN
from imblearn.combine import SMOTETomek,SMOTEENN
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB,CategoricalNB,BernoulliNB 
from sklearn.linear_model import SGDClassifier,LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.metrics import f1_score, make_scorer
from mlxtend.preprocessing import DenseTransformer
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import pandas as pd
import os
from os import listdir
import csv
import matplotlib.pyplot as plt 
from sklearn.metrics import matthews_corrcoef
from scipy.sparse import coo_matrix, hstack,csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


tokenizer = LemmaTokenizer()
token_stop = tokenizer(' '.join(get_stop_words("en")))


def get_df_of_dir(dir_path):
    files = listdir(dir_path)
    df = pd.DataFrame()
    for file in files:
        df_new = pd.read_csv(dir_path + file, engine='python',
                             header=None, encoding='utf-8', error_bad_lines=False, quoting=csv.QUOTE_NONE, delimiter="§")
        df = pd.concat([df, df_new], axis=0)
    return df


def get_df_of_root_dir(path):
    df = pd.DataFrame()
    sub_dirs = [x[0] for x in os.walk(path)]
    sub_dirs.pop(0)
    for sub_dir in sub_dirs:
        df = pd.concat([df, get_df_of_dir(sub_dir + "/")])
    return df


def read_trip_reviews():
    venice_dir = "Venice2"
    stuttgart_dir = "Stuttgart2"
    df_venice = get_df_of_root_dir(venice_dir)
    df_venice['location'] = "venice"
    df_stuttgart = get_df_of_root_dir(stuttgart_dir)
    df_stuttgart['location'] = "stuttgart"
    return pd.concat([df_venice, df_stuttgart], ignore_index=True)

def transver_undersamplingrate(up_rate,down_rate,invert=False):
    inverse_up_rate = 1 - up_rate
    return down_rate * inverse_up_rate + up_rate

def word_count(df):
    df["word_count"] = df['Review'].apply(lambda x: len(x.split()))
    return df
    
def scentenc_count(df):
    df["scentence_count"] = df['Review'].apply(lambda x: len(x.split('.')))
    return df

def word_length(df):
    df["word_length"] = df['Review'].apply(lambda x: sum(len(word) for word in x.split(' ')) / len(x))
    return df

def text_length(df):
    df["text_length"] = df['Review'].apply(lambda x: len(x))
    return df

def captial_letters_count(df):
    df["capital_letters"] = df['Review'].apply(lambda x:  sum(1 for c in x if c.isupper()))
    return df

def sentiment(df):
    analyser = SentimentIntensityAnalyzer()
    df["sentiment"] = df['Review'].apply(lambda x: analyser.polarity_scores(x))
    df["neg_sentiment"] = df["sentiment"].apply(lambda x: x.get("neg"))
    df["pos_sentiment"] = df["sentiment"].apply(lambda x: x.get("pos"))
    df["neu_sentiment"] = df["sentiment"].apply(lambda x: x.get("neu"))
    return df

def first_person_pronouns(df):
    df["fist_person"] = df['Review'].apply(lambda x:  sum(1 for c in x if c in ["I","my","me","mine","we","us","our","ours"]))
    return df

def exclamation_marks(df):
    df["exclamation_marks"] = df['Review'].apply(lambda x:  sum(1 for c in x if c == "!"))
    return df

def positiveWords():
     pass    

def classification_run():
    ### Load Learning Data ###
    df = pd.read_csv("fakenew.csv",  delimiter=";")
    features = df['Review'].to_numpy()
    label = df['Fake1'].to_numpy()

    #### Text Feature Extraction #### 
    df_text = pd.DataFrame()
    df_text['Review'] = df['Review']
    df_text = word_count(df_text)
    df_text = scentenc_count(df_text)
    df_text = captial_letters_count(df_text)
    df_text = word_length(df_text)
    df_text = text_length(df_text)
    df_text = sentiment(df_text)
    df_text = first_person_pronouns(df_text)
    df_text = exclamation_marks(df_text)
    df_text.drop(columns=['Review','sentiment'],inplace=True)

    ### Counter Vectorize / Stop Words / Stemming / Tfidf ###
    args = dict(stop_words=token_stop,
                ngram_range=(1,4),
                strip_accents="unicode",
                tokenizer=LemmaTokenizer(),
                )

    tfidf_vectorizer = TfidfVectorizer(args)
    # tfidf_vectorizer= CountVectorizer(args)
    tfidf_vector = tfidf_vectorizer.fit_transform(features)

    X_train,X_test,y_train,y_test = train_test_split(tfidf_vector, label,test_size=0.2,random_state=42,stratify=label)

    #### Scaling Text Features ####
    text_features = df_text.to_numpy()
    scaler = MinMaxScaler()
    text_features =scaler.fit_transform(text_features)
    
    text_features = csr_matrix(text_features)
    #pdb.set_trace()
    #### Train Test Split ####

    X_train_text,X_test_text,y_train_text,y_test_text = train_test_split(text_features, label,test_size=0.2,random_state=42,stratify=label)

    ##### Over and undersampling #####
    smt = SMOTETomek(n_jobs=8,sampling_strategy=1,random_state=42)
    X, y = smt.fit_resample(X_train,y_train)

    #### Text Feature sampling ####
    smp = RandomUnderSampler(sampling_strategy=1)
    X_train_text_sampl, y_train_text_sampl = smp.fit_resample(X_train_text,y_train_text)

    #### Modelling ####
    mc = make_scorer(matthews_corrcoef)   
    model = ComplementNB()
    model.fit(X, y)

    ##### Text Feature Modelling #####
    text_model = CatBoostClassifier()
    text_model.fit(X_train_text_sampl,y_train_text_sampl)


    #### Model Testing ####
    pred_proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    
    mc = matthews_corrcoef(y_test,pred)
    print("Matthews correlation coefficient: " + str(mc))
    plot_confusion_matrix(model, X_test, y_test) 
    #plt.show()

    #### Text Feature Model Testing ####
    pred_proba_text = text_model.predict_proba(X_test_text)
    pred_text = text_model.predict(X_test_text)
    mc = matthews_corrcoef(y_test_text,pred_text)
    print("Matthews correlation coefficient: " + str(mc))
    plot_confusion_matrix(text_model, X_test_text, y_test_text) 
    #plt.show()

    ### Propability chain ### 
    X_train_chain = np.hstack((model.predict_proba(X_train),text_model.predict_proba(X_train_text)))
    X_test_chain = np.hstack((model.predict_proba(X_test),text_model.predict_proba(X_test_text)))
    chain_modell = CatBoostClassifier()
    chain_modell.fit(X_train_chain,y_train)
    chain_proba = chain_modell.predict(X_test_chain)
    mc = matthews_corrcoef(y_test,chain_proba)
    print("Matthews correlation coefficient chained network: " + str(mc))   

    #### Testing Reviews ####

    # df_reviews = read_trip_reviews()
    # df_venice = df_reviews[df_reviews['location'] == 'venice']
    # df_stuttgart = df_reviews[df_reviews['location'] == 'stuttgart']
    # review_vector_venice = tfidf_vectorizer.transform(df_venice[0].to_numpy())
    # review_vector_stuttgart = tfidf_vectorizer.transform(
    #     df_stuttgart[0].to_numpy())

    # fake_percentage_venice = model.predict(
    #     DenseTransformer().fit_transform(review_vector_venice)).mean() * 100
    # fake_percentage_stuttgart = model.predict(
    #     DenseTransformer().fit_transform(review_vector_stuttgart)).mean()*100

    # print("Fake reviews in Venedik (in Prozent): " + str(fake_percentage_venice))
    # print("Fake reviews in Stuttgart (in Prozent): " +
    #      str(fake_percentage_stuttgart))
    # return {"fake_stuttgart": fake_percentage_stuttgart, "fake_venice": fake_percentage_venice}

print(classification_run())

