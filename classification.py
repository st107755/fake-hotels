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
from imblearn.under_sampling import RandomUnderSampler
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

def append_to_vector(vector,series,scale=True):
    dense = vector.todense()
    array = series.to_numpy().reshape(-1,1)
    scaler = MinMaxScaler()
    array = scaler.fit_transform(array)
    dense = np.append(dense,array,axis=1)
    return csr_matrix(dense)
    
def classification_run():
    ### Load Learning Data ###
    df = pd.read_csv("fakenew.csv",  delimiter=";")
    features = df['Review'].to_numpy()
    label = df['Fake1'].to_numpy()

    #### Extra Feature Extraction #### 
    df = word_count(df)
    df = scentenc_count(df)
    df = captial_letters_count(df)
    df = word_length(df)
    df = text_length(df)
    df = sentiment(df)
    df = first_person_pronouns(df)
    df = exclamation_marks(df)

    #pdb.set_trace()
    ### Counter Vectorize / Stop Words / Stemming / Tfidf ###
    args = dict(stop_words=token_stop,
                ngram_range=(1,4),
                strip_accents="unicode",
                # min_df=2,
                #max_features=2000,
                tokenizer=LemmaTokenizer(),
                )

    tfidf_vectorizer = TfidfVectorizer(args)
    # tfidf_vectorizer= CountVectorizer(args)
    tfidf_vector = tfidf_vectorizer.fit_transform(features)

    tfidf_vector = append_to_vector(tfidf_vector,df["word_count"])
    tfidf_vector = append_to_vector(tfidf_vector,df["scentence_count"])
    tfidf_vector = append_to_vector(tfidf_vector,df["capital_letters"])
    tfidf_vector = append_to_vector(tfidf_vector,df["text_length"])
    tfidf_vector = append_to_vector(tfidf_vector,df["word_length"])
    tfidf_vector = append_to_vector(tfidf_vector,df["neg_sentiment"])
    tfidf_vector = append_to_vector(tfidf_vector,df["pos_sentiment"])
    tfidf_vector = append_to_vector(tfidf_vector,df["neu_sentiment"])
    tfidf_vector = append_to_vector(tfidf_vector,df["fist_person"])
    tfidf_vector = append_to_vector(tfidf_vector,df["exclamation_marks"])


    X_train,X_test,y_train,y_test = train_test_split(tfidf_vector, label,test_size=0.25,random_state=42,stratify=label)

    ##### Over and undersampling #####
    # over = SMOTE(n_jobs=4)
    # # over = ADASYN(n_jobs=4)
    # under = RandomUnderSampler()
    # steps = [('o', over),('u', under)]
    # pipeline = Pipeline(steps=steps)
    # X, y = pipeline.fit_resample(X_train,y_train)
    smt = SMOTETomek(n_jobs=8,sampling_strategy=1,random_state=42)
    X, y = smt.fit_resample(X_train,y_train)
    
    #### Modelling ####
    mc = make_scorer(matthews_corrcoef)
    parameters = {"C": [0.001, 0.01, 0.1, 1.0, 10.0,
                         20.0, 30.0], "loss": ["hinge", "squared_hinge"],
                   "tol": [1e-1, 1e-3, 1e-6]}
    # model = GridSearchCV(
    #      LinearSVC(max_iter=20000), param_grid=parameters, n_jobs=-1, verbose=True, scoring=f1,cv=10
    # )
    model = ComplementNB() 
    # model = BernoulliNB()
    # model = MultinomialNB(fit_prior=False)
    #model = LogisticRegression()
    # model = GaussianNB()
    # model = LinearSVC()
    # model = SGDClassifier()
    #model = KNeighborsClassifier()
    #model = CatBoostClassifier()
    #   
    model.fit(X, y)
    # model.fit(DenseTransformer().fit_transform(X), DenseTransformer().fit_transform(y))

    #### Model Testing ####
    pred = model.predict(X_test)
    # pred = model.predict(DenseTransformer().fit_transform(X_test))
    mc = matthews_corrcoef(y_test,pred)
    # mc = matthews_corrcoef(DenseTransformer().fit_transform(y_test),pred)
    print("Matthews correlation coefficient: " + str(mc))
    plot_confusion_matrix(model, X_test, y_test) 
    plt.show()

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

