from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import pickle
import numpy as np
from datetime import datetime

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


def predict_cov_cos_similarity(prev_sample, prev_similarities, similarities):

    prev_sample = prev_sample.values
    prev_sample_wout_diag = prev_sample[~np.eye(prev_sample.shape[0], dtype=bool)].reshape(prev_sample.shape[0], -1)

    similarities = similarities.values
    similarities_wout_diag = similarities[~np.eye(similarities.shape[0], dtype=bool)].reshape(similarities.shape[0], -1)

    prev_similarities = prev_similarities.values
    prev_similarities_wout_diag = prev_similarities[~np.eye(prev_similarities.shape[0], dtype=bool)].reshape(prev_similarities.shape[0], -1)


    shape = similarities.shape
    prev_similarities_wout_diag = prev_similarities_wout_diag.flatten().reshape(-1,1)
    prev_sample_wout_diag = prev_sample_wout_diag.flatten().reshape(-1,1)
    similarities = similarities.flatten().reshape(-1,1)
    mean_sim = np.mean(prev_similarities_wout_diag)
    sim_prev_demeaned = prev_similarities_wout_diag - mean_sim
    similarities_demeaned = similarities - mean_sim

    lr = LinearRegression()
    lr.fit(sim_prev_demeaned, prev_sample_wout_diag)

    pred = lr.predict(similarities_demeaned)

    pred = pred.reshape(shape)

    diag_prev = prev_sample.diagonal()

    pred[np.diag_indices_from(pred)] = diag_prev

    print(lr.intercept_)
    print(lr.coef_)

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




df_reports = pd.read_csv("data/reports.csv", dtype="string", index_col="date")
df_reports.index = pd.to_datetime(df_reports.index)
df_returns = pd.read_csv("data/stock_returns.csv", index_col="Date")
df_returns.index = pd.to_datetime(df_returns.index)

if os.path.isfile("vectorizer.p"):
    vectorizer = pickle.load(open("vectorizer.p", "rb"))
else:
    vectorizer = train_model(df_reports)
    pickle.dump(vectorizer, open("vectorizer.p", "wb"))

train_first = datetime(year=2005, month=12, day=31)
train_last = datetime(year=2018, month=9, day=30)
test_first = datetime(year=2018, month=12, day=31)
test_last = datetime(year=2020, month=6, day=30)

train_range = pd.date_range(train_first, train_last, freq="Q")
test_range = pd.date_range(test_first, test_last, freq="Q")

for date in train_range:
    returns_stop = date + pd.tseries.offsets.QuarterEnd()

    reports = get_reports_for_date(df_reports, date)
    returns = get_returns_for_period(df_returns, date + pd.DateOffset(days=1), returns_stop)

    similarities = compute_cosine_similarity_matrix(reports)
    cov_matrix = compute_cov_matrix(returns)

    similarities, cov = df_find_intersection([similarities, cov_matrix])


    print(similarities.shape)
    print(cov.shape)


#print(test_range)

'''
reports = get_reports_for_date(df, "31-12-2006")
reports_prev = get_reports_for_date(df, "30-09-2006")

similarities = compute_cosine_similarity_matrix(reports)
similarities_prev = compute_cosine_similarity_matrix(reports_prev)

returns_period = get_returns_for_period(returns, "01-01-2007", "31-03-2007")
cov = compute_cov_matrix(returns_period)

returns_prev_quarter = get_returns_for_period(returns, "01-10-2006", "31-12-2006")
cov_prev_sample = compute_cov_matrix(returns_prev_quarter)

cov, similarities, similarities_prev, cov_prev_sample = df_find_intersection([cov, similarities, similarities_prev, cov_prev_sample])


#returns_prev_quarter, similarities = returns_reports_in_both(returns_prev_quarter, similarities)
#pred_sample = predict_cov_sample(returns_prev_quarter)

pred_sim = predict_cov_cos_similarity(cov_prev_sample,similarities_prev, similarities)
pred_mean = predict_mean(cov_prev_sample)

print(eval_predictions(cov.values, cov_prev_sample.values))
print(eval_predictions(cov.values, pred_sim))
print(eval_predictions(cov.values, pred_mean))

#cov = cov.values.flatten()
#similarities = similarities.values.flatten()

#print(np.corrcoef(cov, similarities))
#print(returns_reports_in_both(cov, similarities)[0].shape)
#print(returns_reports_in_both(cov, similarities)[1].shape)

'''