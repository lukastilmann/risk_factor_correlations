from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
import pandas as pd
import os
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def train_model(df):
    corpus = df.values.flatten()
    notna_corpus = corpus[~pd.isnull(corpus)]
    #vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    bow_vector = vectorizer.fit_transform(notna_corpus)

    print("training lda model")
    lda = LatentDirichletAllocation(n_components=25, random_state=0)

    lda.fit(bow_vector)

    print("finished training lda model")

    return vectorizer, lda

def get_reports_for_date(df, date):

    df_date = df.loc[date]
    df_date = df_date.dropna()

    return df_date

def compute_cosine_similarity_matrix(reports):

    vector_bow = vectorizer.transform(reports)

    vector_lda = lda.transform(vector_bow)

    matrix = cosine_similarity(vector_lda)

    df = pd.DataFrame(matrix, index=reports.index, columns=reports.index)

    return df

def get_returns_for_period(df, start, stop):

    df = df.loc[start:stop]
    #print(df)
    df = df.dropna(axis=1)

    return df


def compute_cov_matrix(returns):

    matrix = np.cov(returns.values.transpose())
    #matrix = np.corrcoef(returns.values.transpose())

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

def calculate_portfolio_var(w, sigma):

    w = np.asmatrix(w)
    sigma = np.asmatrix(sigma)

    print((w.T * sigma) * w)

    return(w.transpose()*sigma*w)


def optimal_portfolio_weights(sigma):

    size = sigma.shape[0]
    sigma = np.asmatrix(sigma)
    #cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0})
    #start_weights = np.full((size, 1), 1 / size)

    one = np.ones((size, 1))
    one = np.asmatrix(one)

    w_star = (np.linalg.inv(sigma) * one) / (one.T * np.linalg.inv(sigma) * one)

    #print(np.sum(w_star))

    return  w_star

def realized_portfolio_returns(returns, w):

    w = np.asarray(w).T

    #print(returns)

    #print(w)

    portfolio_returns = returns *  w

    #print(portfolio_returns)

    #print(portfolio_returns.shape)

    whole_portfolio_returns = np.sum(portfolio_returns, axis=1)

    return_mean = whole_portfolio_returns.mean()

    return_var = whole_portfolio_returns.var()

    sharpe = (return_mean / np.sqrt(return_var)) * np.sqrt(252)

    print(sharpe)





df_reports = pd.read_csv("data/reports.csv", dtype="string", index_col="date")
df_reports.index = pd.to_datetime(df_reports.index)
df_returns = pd.read_csv("data/stock_returns.csv", index_col="Date")
df_returns.index = pd.to_datetime(df_returns.index)

'''

if os.path.isfile("vectorizer_lda_tuple_25.p"):
    vectorizer, lda = pickle.load(open("vectorizer_lda_tuple_25.p", "rb"))
else:
    vectorizer, lda = train_model(df_reports)
    pickle.dump((vectorizer, lda), open("vectorizer_lda_tuple_25.p", "wb"))

train_first = datetime(year=2005, month=12, day=31)
train_last = datetime(year=2018, month=9, day=30)
test_first = datetime(year=2018, month=12, day=31)
test_last = datetime(year=2020, month=6, day=30)

#train_range = pd.date_range(train_first, train_last, freq="Q")
#test_range = pd.date_range(test_first, test_last, freq="Q")

#print(df_reports)

df_reports_train = df_reports.loc[train_first:train_last]

df_reports_test = df_reports.loc[test_first:test_last]

train_range = df_reports_train.index

train_x = []
train_y = []
mean_covs = []
mean_sims = []

for date in train_range:
    returns_stop = date + pd.tseries.offsets.QuarterEnd()

    reports = get_reports_for_date(df_reports, date)
    returns = get_returns_for_period(df_returns, date + pd.DateOffset(days=1), returns_stop)

    similarities = compute_cosine_similarity_matrix(reports)
    cov_matrix = compute_cov_matrix(returns)

    similarities, cov = df_find_intersection([similarities, cov_matrix])

    similarities = similarities.values

    cov = cov.values

    similarities = similarities[~np.eye(similarities.shape[0], dtype=bool)].reshape(-1,1)

    cov = cov[~np.eye(cov.shape[0], dtype=bool)].reshape(-1,1)

    train_x.append(similarities)

    train_y.append(cov)

    mean_covs.append(cov.mean())

    mean_sims.append(similarities.mean())




train_x = np.concatenate(train_x)
train_y = np.concatenate(train_y)

test_range = df_reports_test.index

test_x = []
test_y = []

for date in test_range:
    returns_stop = date + pd.tseries.offsets.QuarterEnd()

    reports = get_reports_for_date(df_reports, date)
    returns = get_returns_for_period(df_returns, date + pd.DateOffset(days=1), returns_stop)

    similarities = compute_cosine_similarity_matrix(reports)
    cov_matrix = compute_cov_matrix(returns)

    similarities, cov = df_find_intersection([similarities, cov_matrix])

    similarities = similarities.values

    cov = cov.values

    similarities = similarities[~np.eye(similarities.shape[0], dtype=bool)].reshape(-1,1)

    cov = cov[~np.eye(cov.shape[0], dtype=bool)].reshape(-1,1)

    test_x.append(similarities)

    test_y.append(cov)


test_x = np.concatenate(test_x)
test_y = np.concatenate(test_y)

x_mean = np.mean(train_x)

#print(x_mean)

train_x = train_x - x_mean
test_x = test_x - x_mean

lr = LinearRegression()

lr.fit(train_x, train_y)

#print(lr.coef_)

pred_y = lr.predict(test_x)

mean_y = np.full(test_y.shape, np.mean(train_y))

'''

returns_period = get_returns_for_period(df_returns, datetime(year=2017, month=1, day=1), datetime(year=2018, month=12, day=31))

returns_out_of_sample = get_returns_for_period(df_returns, datetime(year=2019, month=1, day=1), datetime(year=2019, month=3, day=31))

columns_both = set(returns_period.columns).intersection(returns_out_of_sample.columns)
#returns_sample, returns_out_of_sample = df_find_intersection([returns_period, returns_out_of_sample])

returns_sample = returns_period.loc[:,columns_both]

returns_out_of_sample = returns_out_of_sample.loc[:,columns_both]

cov = predict_cov_sample(returns_sample)

LW = LedoitWolf()

cov_lw = LW.fit(returns_sample).covariance_

#print(cov)

w = optimal_portfolio_weights(cov)

w_lw = optimal_portfolio_weights(cov_lw)

random_weights = np.random.random_sample(w.shape)

w_random = random_weights / random_weights.sum()

w_market_portfolio = np.full(w.shape, 1 / w.shape[0])

realized_portfolio_returns(returns_out_of_sample.values, w_market_portfolio)
realized_portfolio_returns(returns_out_of_sample.values, w_random)
realized_portfolio_returns(returns_out_of_sample, w_lw)
realized_portfolio_returns(returns_out_of_sample.values, w)







#print(calculate_portfolio_var(w, cov))

'''
print("----predictions----")
print("mse: " + str(mean_squared_error(test_y, pred_y)))

print("r2: " + str(r2_score(test_y, pred_y)))

print("----baseline----")
print("mse: " + str(mean_squared_error(test_y, mean_y)))

print("r2: " + str(r2_score(test_y, mean_y)))
'''