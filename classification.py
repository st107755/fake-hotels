import numpy as np
import pandas as pd
import pprint
from itertools import chain
from os import listdir
import csv
import pdb

from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    plot_precision_recall_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)

analyzer = TfidfVectorizer().build_analyzer()
stemmer = EnglishStemmer()


def get_df_of_dir(dir_path: str):
    files = listdir(dir_path)
    df = pd.DataFrame()
    for file in files:
        df_new = pd.read_csv(dir_path + file, engine='python',
                             header=None, encoding='utf-8', error_bad_lines=False, quoting=csv.QUOTE_NONE, delimiter="§")
        df = pd.concat([df, df_new], axis=0)
    return df


def read_trip_reviews():
    venice_dir = "Al_Ponte_Antico_Hotel/en/"
    stuttgart_dir = "Mövenpick_Hotel_Stuttgart_Airport/en/"
    df_venice = get_df_of_dir(venice_dir)
    df_venice['location'] = "venice"
    df_stuttgart = get_df_of_dir(stuttgart_dir)
    df_stuttgart['location'] = "stuttgart"
    return pd.concat([df_venice, df_stuttgart], ignore_index=True)


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


df = pd.read_csv("fakenew.csv",  delimiter=";")

features = df['Review'].to_numpy()
label = df['Fake1'].to_numpy()

### Counter Vectorize / Stop Words / Stemming / Tfidf ###
optimal_args = dict(stop_words=get_stop_words(
    "en"), analyzer=stemmed_words,  min_df=2)

tfidf_vectorizer = TfidfVectorizer(optimal_args)
tfidf_vector = tfidf_vectorizer.fit_transform(features)


random_seed = np.random.randint(1, len(label))
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_vector.toarray(),
    label,
    test_size=0.2,
    random_state=random_seed,
    shuffle=True,
    stratify=label,
)

#### Modelling ####
clf = LinearSVC()
model = clf.fit(X_train, y_train)
prediction = model.predict(X_test)


#### Testing Reviews ####
df_reviews = read_trip_reviews()
df_venice = df_reviews[df_reviews['location'] == 'venice']
df_stuttgart = df_reviews[df_reviews['location'] == 'stuttgart']
review_vector_venice = tfidf_vectorizer.transform(df_venice[0].to_numpy())
review_vector_stuttgart = tfidf_vectorizer.transform(
    df_stuttgart[0].to_numpy())

fake_percentage_venice = model.predict(review_vector_venice).mean() * 100
fake_percentage_stuttgart = model.predict(review_vector_stuttgart).mean()*100
print("Fake reviews in Venedik (in Prozent): " + str(fake_percentage_venice))
print("Fake reviews in Stuttgart (in Prozent): " + str(fake_percentage_stuttgart))
# pdb.set_trace()


#report = classification_report(y_test, prediction, output_dict=True)
#score = model.score(X_test, y_test)
# pprint.pprint(report)
