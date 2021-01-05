from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

def get_reports_for_date(df, date):

    df_date = df.loc[date]
    df_date = df_date.dropna()

    return df_date

def compute_cosine_similarity_matrix(reports):

    vector = vectorizer.transform(reports)

    matrix = cosine_similarity(vector)

    df = pd.DataFrame(matrix, index=reports.index, columns=reports.index)

    return df

def get_returns_for_period(df, start, stop):

    df = df.loc[start:stop]
    #print(df)
    df = df.dropna(axis=1)

    return df


def compute_cov_matrix(returns):

    matrix = np.cov(returns.values.transpose())

    df = pd.DataFrame(matrix, index=returns.columns, columns=returns.columns)

    return df

def df_find_intersection(list_dfs):

    list_columns = []

    for df in list_dfs:
        list_columns.append(set(df.columns))

    in_all = set.intersection(*list_columns)

    list_dfs_new = []

    for df in list_dfs:
        list_dfs_new.append(df.loc[in_all,in_all])

    return list_dfs_new


def predict_cov_sample(prev_sample):

    return compute_cov_matrix(prev_sample).values


def predict_cov_cos_similarity(prev_sample, similarities):

    prev_sample = prev_sample.values

    similarities = similarities.values

    mean_cov = np.mean(prev_sample)

    sigma_cov = np.std(prev_sample)

    mean_sim = np.mean(similarities)

    sigma_sim = np.std(similarities)

    c = (mean_sim - similarities) / sigma_sim

    #print(c)

    pred = mean_cov - c * sigma_cov

    return pred

def predict_mean(prev_sample):

    cov_mat = prev_sample.values
    without_diag = cov_mat[~np.eye(cov_mat.shape[0], dtype=bool)].reshape(cov_mat.shape[0], -1)
    mean_cov = np.mean(without_diag)
    mean_cov = np.full(cov_mat.shape, mean_cov)
    mean_variance = np.mean(np.diagonal(cov_mat))
    np.fill_diagonal(mean_cov, mean_variance)

    return mean_cov

def eval_predictions(true, pred):
    return (np.square(true - pred)).mean(axis=None)




df = pd.read_csv("data/reports.csv", dtype="string", index_col="date")
df.index = pd.to_datetime(df.index)

if os.path.isfile("vectorizer.p"):
    vectorizer = pickle.load(open("vectorizer.p", "rb"))
else:
    vectorizer = train_model(df)
    pickle.dump(vectorizer, open("vectorizer.p", "wb"))

reports = get_reports_for_date(df, "31-12-2006")
returns = pd.read_csv("data/stock_returns.csv", index_col="Date")
returns.index = pd.to_datetime(returns.index)

similarities = compute_cosine_similarity_matrix(reports)

returns_period = get_returns_for_period(returns, "01-01-2007", "31-03-2007")
cov = compute_cov_matrix(returns_period)

returns_prev_quarter = get_returns_for_period(returns, "01-10-2006", "31-12-2006")
cov_prev_sample = compute_cov_matrix(returns_prev_quarter)

cov, similarities, cov_prev_sample = df_find_intersection([cov, similarities, cov_prev_sample])


#returns_prev_quarter, similarities = returns_reports_in_both(returns_prev_quarter, similarities)
#pred_sample = predict_cov_sample(returns_prev_quarter)

pred_sim = predict_cov_cos_similarity(cov_prev_sample, similarities)
pred_mean = predict_mean(cov_prev_sample)

print(eval_predictions(cov.values, cov_prev_sample.values))
print(eval_predictions(cov.values, pred_sim))
print(eval_predictions(cov.values, pred_mean))

#cov = cov.values.flatten()
#similarities = similarities.values.flatten()

#print(np.corrcoef(cov, similarities))
#print(returns_reports_in_both(cov, similarities)[0].shape)
#print(returns_reports_in_both(cov, similarities)[1].shape)

