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
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN, KMeansSMOTE, BorderlineSMOTE, SMOTENC
from catboost import CatBoostClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
import scipy.stats as stats
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.metrics import f1_score, make_scorer
from mlxtend.preprocessing import DenseTransformer
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, ttest_rel
import scipy
from os import listdir
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from scipy.sparse import coo_matrix, hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


tokenizer = LemmaTokenizer()
token_stop = tokenizer(' '.join(get_stop_words("en")))
pd.set_option('mode.chained_assignment', None)


def get_df_of_dir(dir_path):
    files = listdir(dir_path)
    df = pd.DataFrame()
    for file in files:
        df_new = pd.read_csv(dir_path + file, engine='python',
                             header=None, encoding='utf-8', error_bad_lines=False, quoting=csv.QUOTE_NONE, delimiter="§", names=["Review"])
        df = pd.concat([df, df_new], axis=0)
    return df


def get_df_of_root_dir(path):
    df = pd.DataFrame()
    sub_dirs = [x[0] for x in os.walk(path)]
    sub_dirs.pop(0)
    for sub_dir in sub_dirs:
        sub = get_df_of_dir(sub_dir + "/")
        # pdb.set_trace()
        sub["hotel"] = str(sub_dir).split("/")[1]
        df = pd.concat([df, sub])
    return df


def read_trip_reviews():
    venice_dir = "Venice2"
    stuttgart_dir = "Stuttgart2"
    df_venice = get_df_of_root_dir(venice_dir)
    df_venice['location'] = "venice"
    df_stuttgart = get_df_of_root_dir(stuttgart_dir)
    df_stuttgart['location'] = "stuttgart"
    return pd.concat([df_venice, df_stuttgart], ignore_index=True)


def transver_undersamplingrate(up_rate, down_rate, invert=False):
    inverse_up_rate = 1 - up_rate
    return down_rate * inverse_up_rate + up_rate


def word_count(df):
    df["word_count"] = df['Review'].apply(lambda x: len(x.split()))
    return df


def scentenc_count(df):
    df["scentence_count"] = df['Review'].apply(lambda x: len(x.split('.')))
    return df


def word_length(df):
    df["word_length"] = df['Review'].apply(
        lambda x: sum(len(word) for word in x.split(' ')) / len(x))
    return df


def text_length(df):
    df["text_length"] = df['Review'].apply(lambda x: len(x))
    return df


def captial_letters_count(df):
    df["capital_letters"] = df['Review'].apply(
        lambda x:  sum(1 for c in x if c.isupper()))
    return df


def sentiment(df):
    analyser = SentimentIntensityAnalyzer()
    df["sentiment"] = df['Review'].apply(lambda x: analyser.polarity_scores(x))
    df["neg_sentiment"] = df["sentiment"].apply(lambda x: x.get("neg"))
    df["pos_sentiment"] = df["sentiment"].apply(lambda x: x.get("pos"))
    df["neu_sentiment"] = df["sentiment"].apply(lambda x: x.get("neu"))
    return df


def first_person_pronouns(df):
    df["fist_person"] = df['Review'].apply(lambda x:  sum(
        1 for c in x if c in ["I", "my", "me", "mine", "we", "us", "our", "ours"]))
    return df


def exclamation_marks(df):
    df["exclamation_marks"] = df['Review'].apply(
        lambda x:  sum(1 for c in x if c == "!"))
    return df


def append_to_vector(vector, series, scale=True):
    dense = vector.todense()
    array = series.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    array = scaler.fit_transform(array)
    dense = np.append(dense, array, axis=1)
    return csr_matrix(dense)


def append_text_features_to_vector(vector, df):
    vector = append_to_vector(vector, df["word_count"])
    vector = append_to_vector(vector, df["scentence_count"])
    vector = append_to_vector(vector, df["capital_letters"])
    vector = append_to_vector(vector, df["text_length"])
    vector = append_to_vector(vector, df["word_length"])
    vector = append_to_vector(vector, df["neg_sentiment"])
    vector = append_to_vector(vector, df["pos_sentiment"])
    vector = append_to_vector(vector, df["neu_sentiment"])
    vector = append_to_vector(vector, df["fist_person"])
    vector = append_to_vector(vector, df["exclamation_marks"])
    return vector


def extract_text_features(df):
    df = word_count(df)
    df = scentenc_count(df)
    df = captial_letters_count(df)
    df = word_length(df)
    df = text_length(df)
    df = sentiment(df)
    df = first_person_pronouns(df)
    df = exclamation_marks(df)
    return df

def total_fake_percentage(df,model,vectorizer):
    df = extract_text_features(df)
    vector = vectorizer.transform(df['Review'].to_numpy())
    vector = append_text_features_to_vector(vector, df)
    return model.predict(vector).mean()


def hotel_fake_percentage_list(df, model, vectorizer):
    percentages = []
    for hotel in df.hotel.unique():
        df_hotel = df[df['hotel'] == hotel]
        df_hotel = extract_text_features(df_hotel)
        # pdb.set_trace()
        vector = vectorizer.transform(df_hotel['Review'].to_numpy())
        vector = append_text_features_to_vector(vector, df_hotel)
        fake_percentage = model.predict(vector).mean() * 100
        percentages.append(fake_percentage)
    return percentages

def random_fake_percentage_list(df,model,vectorizer,parts=20):
    percentages = []
    df = shuffle(df)
    df_length = len(df.index)
    part_length = df_length/parts
    for i in range(0,parts):
        start = int(i * part_length)
        end = int((i+1) * part_length)
        part = df[start:end]
        part = extract_text_features(part)
        vector = vectorizer.transform(part['Review'].to_numpy())
        vector = append_text_features_to_vector(vector, part)
        fake_percentage = model.predict(vector).mean() * 100
        percentages.append(fake_percentage)
    return percentages

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1)  # calculate F test statistic
    dfn = x.size-1  # define degrees of freedom numerator
    dfd = y.size-1  # define degrees of freedom denominator
    p = 1-scipy.stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
    return f, p


def super_sample(samples, i=100):
    samples = np.array(samples)
    super_sample = []
    for j in range(0, i):
        samples_for_mean = []
        for k in range(0, len(samples)):
            samples_for_mean.append(np.random.choice(samples, 1))
        super_sample.append(np.array(samples_for_mean).mean())
    return super_sample

def average(lst): 
    return sum(lst) / len(lst) 


def classification_run():
    ### Load Learning Data ###
    df = pd.read_csv("fakenew.csv",  delimiter=";")
    features = df['Review'].to_numpy()
    label = df['Fake1'].to_numpy()

    #### Extra Feature Extraction ####
    df = extract_text_features(df)

    ### Counter Vectorize / Stop Words / Stemming / Tfidf ###
    args = dict(stop_words=token_stop,
                ngram_range=(1, 4),
                strip_accents="unicode",
                tokenizer=LemmaTokenizer(),
                )

    tfidf_vectorizer = TfidfVectorizer(args)
    tfidf_vector = tfidf_vectorizer.fit_transform(features)

    tfidf_vector = append_text_features_to_vector(tfidf_vector, df)

    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_vector, label, test_size=0.25, random_state=42, stratify=label)

    ##### Over and undersampling #####
    over = SMOTE(n_jobs=4, random_state=370, sampling_strategy=1.0)
    X, y = over.fit_resample(X_train, y_train)

    #### Modelling ####
    mc = make_scorer(matthews_corrcoef)
    parameters = {"C": [0.001, 0.01, 0.1, 1.0, 10.0,
                        20.0, 30.0], "loss": ["hinge", "squared_hinge"],
                  "tol": [1e-1, 1e-3, 1e-6]}
    # model = GridSearchCV(
    #      LinearSVC(max_iter=20000), param_grid=parameters, n_jobs=-1, verbose=True, scoring=mc,cv=10
    # )
    # model = LinearSVC()
    model = ComplementNB()
    # model = LogisticRegression()
    # model = SGDClassifier()

    model.fit(X, y)

    #### Model Testing ####
    pred = model.predict(X_test)
    mc = matthews_corrcoef(y_test, pred)
    print("Matthews correlation coefficient: " + str(mc))
    plot_confusion_matrix(model, X_test, y_test)
    # plt.show()

    #### Testing Reviews ####

    df_reviews = read_trip_reviews()
    df_venice = df_reviews[df_reviews['location'] == 'venice']
    df_stuttgart = df_reviews[df_reviews['location'] == 'stuttgart']

    fake_venice = total_fake_percentage(df_venice, model, tfidf_vectorizer)
    fake_stuttgart = total_fake_percentage(df_stuttgart, model, tfidf_vectorizer)
    # stats.probplot(fake_stuttgart, dist="norm", plot=plt)
    # plt.show()
    venice_fake = fake_venice * len(df_venice.index)
    venice_non_fake = (1-fake_venice) * len(df_venice.index)
    stuttgart_fake = fake_stuttgart * len(df_stuttgart.index)
    stuttgart_non_fake = (1-fake_stuttgart) * len(df_stuttgart.index)
    oddsratio, pvalue = stats.fisher_exact([[venice_fake, venice_non_fake], [stuttgart_fake, stuttgart_non_fake]])
    print(pvalue)
    # print("Venice Fake: " + str())
    # print("Venice Non Fake: " + str())
    # print("Stuttgart Fake: " + str())
    # print("Stuttgart Non Fake: " + str())
    # print(ttest_ind(fake_venice, fake_stuttgart))
    # print(ttest_rel(fake_venice, fake_stuttgart))
    # print(mannwhitneyu(fake_venice, fake_stuttgart))


classification_run()
