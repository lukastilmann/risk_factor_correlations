from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import pickle
import numpy as np

def train_model(df):
    corpus = df.values.flatten()
    notna_corpus = corpus[~pd.isnull(corpus)]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    vectorizer = vectorizer.fit(notna_corpus)

    return vectorizer


df = pd.read_csv("data/reports.csv", dtype="string")

if os.path.isfile("vectorizer.p"):
    vectorizer = pickle.load(open("vectorizer.p", "rb"))
else:
    vectorizer = train_model(df)
    pickle.dump(vectorizer, open("vectorizer.p", "wb"))
