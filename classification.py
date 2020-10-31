import numpy as np
import pandas as pd
import pprint
from itertools import chain
from os import listdir
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


def read_trip_reviews():
    venice_dir = "Al_Ponte_Antico_Hotel/en/"
    venice_files = listdir(venice_dir)
    df_venice = pd.DataFrame()
    for file in venice_files:
        print(file)
        df = pd.read_csv(venice_dir + file, engine='python')
        df_venice = pd.concat([df_venice, df], axis=1)
    pdb.set_trace()
    df_stuttgart = pd.read_csv("Mövenpick_Hotel_Stuttgart_Airport")


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


df = pd.read_csv("fakenew.csv",  delimiter=";")

features = df['Review'].to_numpy()
label = df['Fake1'].to_numpy()

### Counter Vectorize / Stop Words / Stemming / Tfidf ###
optimal_args = dict(stop_words=get_stop_words(
    "en"), analyzer=stemmed_words, min_df=2,)

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
# clf = ComplementNB()
clf = LinearSVC()
model = clf.fit(X_train, y_train)
prediction = model.predict(X_test)


#### Testing Reviews ####
read_trip_reviews()

report = classification_report(y_test, prediction, output_dict=True)
score = model.score(X_test, y_test)
pprint.pprint(report)
