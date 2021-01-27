from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
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

    df_date = df.loc[date:date]
    df_date = df_date.dropna(axis=1)

    return df_date

def lda_features(reports, vectorizer, lda):

    if isinstance(reports, pd.DataFrame):
        reports = reports.iloc[0]

    vector_bow = vectorizer.transform(reports)

    vector_lda = lda.transform(vector_bow)

    #print(vector_lda.shape)

    return vector_lda

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

def find_column_intersection(list_dfs):

    list_columns = []

    for df in list_dfs:
        #print(df.columns)
        list_columns.append(set(df.columns))

    in_all = set.intersection(*list_columns)

    list_dfs_new = []

    #print(in_all)

    for df in list_dfs:
        list_dfs_new.append(df[in_all])

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

    #print((w.T * sigma) * w)

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

    portfolio_value = [100]

    for r in whole_portfolio_returns:

        portfolio_value.append(portfolio_value[-1] * (1 + r))


    print(sharpe)

    return portfolio_value

def exp_dist(x_1, x_2):
    dist = x_1 - x_2
    sim = np.exp(-1 * np.square(dist))

    return sim

def get_similarities_cov(mat, feature_data, sim_function):

    flat_upper = mat[np.triu_indices(mat.shape[0], k=1)]

    n = feature_data.shape[0]

    similarities = []

    for i in range(n):
        features_i = feature_data[i,:]
        for j in range(i+1,n):
            features_j = feature_data[j,:]
            pairwise_sim = sim_function(features_i, features_j)
            #print(pairwise_sim)
            similarities.append(pairwise_sim)

    similarities = np.stack(similarities)

    return (similarities, flat_upper)

def predict_covariance_matrix_model(model, scaler, feature_data, mean_var):

    sim_measure = lambda x_1,x_2 : model.predict(scaler.transform(exp_dist(x_1,x_2).reshape(1, -1)))

    matrix = pairwise_distances(feature_data, metric=sim_measure)

    np.fill_diagonal(matrix, mean_var)

    return matrix


df_reports = pd.read_csv("data/reports.csv", dtype="string", index_col="date")
df_reports.index = pd.to_datetime(df_reports.index)
df_returns = pd.read_csv("data/stock_returns.csv", index_col="Date")
df_returns.index = pd.to_datetime(df_returns.index)



if os.path.isfile("vectorizer_lda_tuple_25.p"):
    vectorizer, lda = pickle.load(open("vectorizer_lda_tuple_25.p", "rb"))
else:
    vectorizer, lda = train_model(df_reports)
    pickle.dump((vectorizer, lda), open("vectorizer_lda_tuple.p_25", "wb"))


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

    returns, reports = find_column_intersection([returns, reports])

    cov = compute_cov_matrix(returns)

    reports_features = lda_features(reports, vectorizer, lda)

    sim_pairwise, cov_udig = get_similarities_cov(cov, reports_features, exp_dist)


    train_x.append(sim_pairwise)

    train_y.append(cov_udig)

    mean_covs.append(cov.diagonal().mean())


train_x = np.stack(train_x)
train_y = np.stack(train_y)

'''
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



returns_sample = get_returns_for_period(df_returns, datetime(year=2018, month=1, day=1), datetime(year=2018, month=12, day=31))

returns_out_of_sample = get_returns_for_period(df_returns, datetime(year=2019, month=1, day=1), datetime(year=2019, month=3, day=31))

reports_sample = get_reports_for_date(df_reports, datetime(year=2017, month=12, day=31))

reports_out_of_sample = get_reports_for_date(df_reports, datetime(year=2018, month=12, day=31))

returns_sample, returns_out_of_sample, reports_sample, reports_out_of_sample = find_column_intersection([returns_sample, returns_out_of_sample, reports_sample, reports_out_of_sample])

cov = predict_cov_sample(returns_sample)
cov_random = np.random.random_sample(cov.shape)

reports_features_sample = lda_features(reports_sample, vectorizer, lda)

#print(reports_features_sample)

reports_features_out_of_sample = lda_features(reports_out_of_sample, vectorizer, lda)

sim_sample, cov_sample = get_similarities_cov(cov, reports_features_sample, exp_dist)

scaler = StandardScaler()

sim_sample = scaler.fit_transform(sim_sample)

#print(sim_sample)

#lr = LinearRegression()
lr = ElasticNetCV()



lr.fit(sim_sample, cov_sample)

print(lr.intercept_)
print(lr.coef_)

sample_mean_var = np.diag(cov).mean()

cov_lda_model = predict_covariance_matrix_model(lr, scaler, reports_features_out_of_sample, sample_mean_var)



LW = LedoitWolf()

cov_lw = LW.fit(returns_sample).covariance_

#print(cov)

w_sample = optimal_portfolio_weights(cov)

w_lw = optimal_portfolio_weights(cov_lw)

random_weights = np.random.random_sample(w_sample.shape)

w_random = optimal_portfolio_weights(cov_random)

w_market_portfolio = np.full(w_sample.shape, 1 / w_sample.shape[0])

w_model = optimal_portfolio_weights(cov_lda_model)

r_market = realized_portfolio_returns(returns_out_of_sample.values, w_market_portfolio)
r_random = realized_portfolio_returns(returns_out_of_sample.values, w_random)
r_lw = realized_portfolio_returns(returns_out_of_sample, w_lw)
r_sample = realized_portfolio_returns(returns_out_of_sample.values, w_sample)
r_model = realized_portfolio_returns(returns_out_of_sample, w_model)

x = range(len(r_market))

plt.plot(x, r_market, label="market")
plt.plot(x, r_random, label="random")
plt.plot(x, r_lw, label="ledoit wolf")
plt.plot(x, r_sample, label="sample")
plt.plot(x, r_model, label="lda model")

plt.legend()

plt.savefig("returns.png")




'''

print("----predictions----")
print("mse: " + str(mean_squared_error(test_y, pred_y)))

print("r2: " + str(r2_score(test_y, pred_y)))

print("----baseline----")
print("mse: " + str(mean_squared_error(test_y, mean_y)))

print("r2: " + str(r2_score(test_y, mean_y)))
'''