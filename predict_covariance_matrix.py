from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import pickle
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import QuarterBegin, QuarterEnd, DateOffset
from pandas import Timedelta
import matplotlib.pyplot as plt
import csv


def train_tfidf_model(df, train_last, idf=True):
    corpus = df.loc[:train_last].values.flatten()
    notna_corpus = corpus[~pd.isnull(corpus)]
    vectorizer = TfidfVectorizer(stop_words="english", strip_accents="unicode", min_df=10, use_idf=idf)

    if idf:
        print("training tfidf model.")
    else:
        print("training tf model")

    vectorizer = vectorizer.fit(notna_corpus)

    return vectorizer


def train_lda_model(df, train_last, n_dims):
    corpus = df.loc[:train_last].values.flatten()
    notna_corpus = corpus[~pd.isnull(corpus)]
    vectorizer = CountVectorizer(stop_words="english", min_df=10)
    bow_vector = vectorizer.fit_transform(notna_corpus)

    print("training lda model with {} dimensions.".format(str(n_dims)))
    lda = LatentDirichletAllocation(n_components=n_dims, random_state=0)

    lda.fit(bow_vector)

    print("finished training lda model.")

    return vectorizer, lda


def train_svd_model(df, train_last, n_dims):
    corpus = df.loc[:train_last].values.flatten()
    notna_corpus = corpus[~pd.isnull(corpus)]
    vectorizer = CountVectorizer(stop_words="english", min_df=10)
    bow_vector = vectorizer.fit_transform(notna_corpus)

    print("training svd model with {} dimensions.".format(str(n_dims)))
    svd = TruncatedSVD(n_components=n_dims, random_state=0)

    svd.fit(bow_vector)

    print("finished training svd model.")

    return vectorizer, svd


def get_reports_for_date(df, date):
    df_date = df.loc[date:date]
    df_date = df_date.dropna(axis=1)

    return df_date


def topic_model_features(reports, vectorizer, model):
    if isinstance(reports, pd.DataFrame):
        reports = reports.iloc[0]

    vector_bow = vectorizer.transform(reports)

    vector_topics = model.transform(vector_bow)

    return vector_topics


def tfidf_features(reports, vectorizer):
    if isinstance(reports, pd.DataFrame):
        reports = reports.iloc[0]

    return vectorizer.transform(reports)


def get_returns_for_period(df, start, stop):
    df = df.loc[start:stop]
    # print(df)
    df = df.dropna(axis=1)

    return df


def compute_cov_matrix(returns, corr=False):
    if corr:
        matrix = np.corrcoef(returns.values.transpose())
    else:
        matrix = np.cov(returns.values.transpose())
    df = pd.DataFrame(matrix, index=returns.columns, columns=returns.columns)

    return df


def find_column_intersection(list_dfs):
    list_columns = []

    for df in list_dfs:
        list_columns.append(set(df.columns))

    in_all = set.intersection(*list_columns)

    list_dfs_new = []

    for df in list_dfs:
        list_dfs_new.append(df[in_all])

    return list_dfs_new


def predict_cov_sample(prev_sample, corr=False):

    return compute_cov_matrix(prev_sample, corr).values


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

    return (w.transpose() * sigma * w)


def optimal_portfolio_weights(sigma):
    size = sigma.shape[0]
    sigma = np.asmatrix(sigma)
    one = np.ones((size, 1))
    one = np.asmatrix(one)
    w_star = (np.linalg.inv(sigma) * one) / (one.T * np.linalg.inv(sigma) * one)

    return w_star


def realized_portfolio_returns(returns, w_mat, returns_previous):
    w = np.asarray(w_mat).T
    portfolio_returns = returns * w
    whole_portfolio_returns = np.sum(portfolio_returns, axis=1)

    return_mean = whole_portfolio_returns.mean()
    return_var = whole_portfolio_returns.var()
    sharpe = (return_mean / np.sqrt(return_var)) * np.sqrt(252)

    portfolio_returns = returns_previous
    for r in whole_portfolio_returns:
        portfolio_returns.append(portfolio_returns[-1] * (1 + r))

    print("sharpe ratio: " + str(sharpe))
    print("r var: " + str(return_var))
    print("-----")

    return portfolio_returns, return_var


def exp_dist(x_1, x_2):
    dist = np.absolute(x_1 - x_2)
    sim = np.exp(-1 * np.square(dist))

    return sim


def get_similarities_cov(mat, feature_data, sim_function, feature_wise, standardize):
    flat_upper = mat[np.triu_indices(mat.shape[0], k=1)]

    if standardize:
        cov_mean = flat_upper.mean()
        cov_mat = flat_upper - cov_mean
    else:
        cov_mat = flat_upper

    n = feature_data.shape[0]

    if feature_wise:

        similarities = []

        for i in range(n):
            features_i = feature_data[i, :]
            for j in range(i + 1, n):
                features_j = feature_data[j, :]
                pairwise_sim = sim_function(features_i, features_j)
                similarities.append(pairwise_sim)

        similarities = np.stack(similarities)

    else:
        similarities = sim_function(feature_data)
        similarities = similarities[np.triu_indices(similarities.shape[0], k=1)]

    return similarities, cov_mat


def predict_covariance_matrix_model(model, scaler, feature_data, mean_var, mean_cov, feature_wise, add_mean):
    # for feature wise:
    if feature_wise:
        sim_measure = lambda x_1, x_2: model.predict(scaler.transform(exp_dist(x_1, x_2).reshape(1, -1)))

    # for pairwise
    else:
        sim_measure = lambda x_1, x_2: model.predict(
            scaler.transform(cosine_similarity(x_1.reshape(1, -1), x_2.reshape(1, -1))))

    matrix = pairwise_distances(feature_data, metric=sim_measure)
    if add_mean:
        matrix = matrix + mean_cov
    np.fill_diagonal(matrix, mean_var)

    return matrix


def corr_matrix_to_cov_matrix(cor_matrix, diag):

    v = np.asmatrix(diag)
    v_matrix = np.sqrt(np.matmul(v.T, v))
    cov_est = np.multiply(cor_matrix, v_matrix)

    return cov_est


def predict_correlation_matrix_model(model, scaler, feature_data, mean_cor, feature_wise, add_mean, cov_mat):
    # for feature wise:
    if feature_wise:
        sim_measure = lambda x_1, x_2: model.predict(scaler.transform(exp_dist(x_1, x_2).reshape(1, -1)))

    # for pairwise
    else:
        sim_measure = lambda x_1, x_2: model.predict(
            scaler.transform(cosine_similarity(x_1.reshape(1, -1), x_2.reshape(1, -1))))

    matrix = pairwise_distances(feature_data, metric=sim_measure)
    if add_mean:
        matrix = matrix + mean_cor
    np.fill_diagonal(matrix, 1)
    diag = np.diag(cov_mat)
    matrix = corr_matrix_to_cov_matrix(matrix, diag)

    return matrix


def constant_covariance_model(sample_cov, mean_corr):

    diag = np.diag(sample_cov)
    cor_matrix = np.full(sample_cov.shape, mean_corr)
    np.fill_diagonal(cor_matrix, 1)
    cov_est = corr_matrix_to_cov_matrix(cor_matrix, diag)

    return cov_est


# parameters:
time_horizon_quarters = 1
model = "tfidf"
idf = False
feature_wise = False
n_dims = 5
with_intercept = False
standardize_cov_matrix = True
predict_corr = True
trial_name = "tfidf_1Q_corr_no_standardize"

# loading data
df_reports = pd.read_csv("data/reports_with_duplicates_final.csv", dtype="string", index_col="date")
df_reports.index = pd.to_datetime(df_reports.index)
df_returns = pd.read_csv("data/stock_returns.csv", index_col="Date")
df_returns.index = pd.to_datetime(df_returns.index)

# defining train test split
train_first = datetime(year=2005, month=12, day=31)
train_last = datetime(year=2018, month=9, day=30)

if model == "tfidf":
    # string with pickle name for saved model
    if idf:
        weighting = "tfidf"
    else:
        weighting = "tf"
    pickle_name = "vectorizer_{}.p".format(weighting)
    # checking if model has been saved
    if os.path.isfile(pickle_name):
        vectorizer = pickle.load(open(pickle_name, "rb"))
    else:
        vectorizer = train_tfidf_model(df_reports, train_last, idf)
        pickle.dump(vectorizer, open(pickle_name, "wb"))

if model in ["svd", "lda"]:
    # string with name for pickle to save model to
    pickle_name = "vectorizer_{model}_tuple_{dim}.p".format(model=model, dim=n_dims)

    if os.path.isfile(pickle_name):
        vectorizer, topic_model = pickle.load(open(pickle_name, "rb"))
    else:
        if model == "svd":
            vectorizer, topic_model = train_svd_model(df_reports, train_last, n_dims)
        else:
            vectorizer, topic_model = train_lda_model(df_reports, train_last, n_dims)
        pickle.dump((vectorizer, topic_model), open(pickle_name, "wb"))

# loading reports for training set
df_reports_train = df_reports.loc[train_first:train_last]
train_range = df_reports_train.index

train_x = []
train_y = []
##mean_covs = []
mean_sims = []

for date in train_range:

    print(date)

    # if datetime(year=2008, month=6, day=30) <= date <= datetime(year=2009, month=3, day=31):
    #    continue
    # returns_stop = date + pd.tseries.offsets.QuarterEnd()
    returns_stop = date + Timedelta(weeks=52)

    reports = get_reports_for_date(df_reports, date)
    returns = get_returns_for_period(df_returns, date + pd.DateOffset(days=1), returns_stop)

    returns, reports = find_column_intersection([returns, reports])

    cov = predict_cov_sample(returns)
    cor = predict_cov_sample(returns, True)

    if model == "tfidf":
        reports_features = tfidf_features(reports, vectorizer)
    if model in ["svd", "lda"]:
        reports_features = topic_model_features(reports, vectorizer, topic_model)

    if predict_corr:
        est_target = cor
    else:
        est_target = cov

    if feature_wise:
        sim, est_target_upper_dig = get_similarities_cov(est_target, reports_features, exp_dist, feature_wise=True, standardize=standardize_cov_matrix)
    else:
        sim, est_target_upper_dig = get_similarities_cov(est_target, reports_features, cosine_similarity, feature_wise=False, standardize=standardize_cov_matrix)

    train_x.append(sim)
    train_y.append(est_target_upper_dig)
    #mean_covs.append(cov.diagonal().mean())

    #print(cor.diagonal().mean())
    cor_upper = cor[np.triu_indices(cor.shape[0], k=1)]
    print(cor_upper.mean())

# creating the training data
train_x = np.concatenate(train_x, axis=0)
train_y = np.concatenate(train_y, axis=0)

# standardizing the data
scaler = StandardScaler()
if not feature_wise:
    train_x = train_x.reshape(-1, 1)
train_x = scaler.fit_transform(train_x)

# regression model for prediction
lr = LinearRegression(fit_intercept=with_intercept)
lr.fit(train_x, train_y)

# mean variance in the sample, as model doesn't predict variance
#sample_mean_var = np.mean(mean_covs)

# the following is to test to trained model
total_quarters = 8

test_intervals = int(total_quarters / time_horizon_quarters)

r_equal = [100]
r_constant = [100]
r_sample = [100]
r_lw = [100]
r_model = [100]
r_combined = [100]

frob_rows = [["period", "equal", "constant", "sample", "lw", "model", "combined"]]
var_rows = [["period", "equal", "constant", "sample", "lw", "model", "combined"]]

for i in range(test_intervals):

    sample_start = datetime(year=2018, month=1, day=1) + QuarterBegin(startingMonth=1, n=i * time_horizon_quarters)
    sample_stop = sample_start + DateOffset(years=1) - DateOffset(days=1)

    out_of_sample_start = sample_stop + DateOffset(days=1)
    out_of_sample_stop = out_of_sample_start + QuarterEnd(startingMonth=3, n=time_horizon_quarters)

    # creating the reports and returns for the test.
    # Includes sample (previous time frame used for empirical estimation) and the test set
    returns_sample = get_returns_for_period(df_returns, sample_start, sample_stop)
    returns_out_of_sample = get_returns_for_period(df_returns, out_of_sample_start, out_of_sample_stop)

    reports_sample = get_reports_for_date(df_reports, sample_start - Timedelta(days=1))
    reports_out_of_sample = get_reports_for_date(df_reports, out_of_sample_start - Timedelta(days=1))

    print("-----------------new test period-----------------")
    print("reports for: " + str(out_of_sample_start - Timedelta(days=1)))

    returns_sample, returns_out_of_sample, reports_sample, reports_out_of_sample = find_column_intersection(
        [returns_sample, returns_out_of_sample, reports_sample, reports_out_of_sample])

    # feature engineer for the time frame to predict
    if model == "tfidf":
        reports_features_out_of_sample = tfidf_features(reports_out_of_sample, vectorizer)
    if model in ["svd", "lda"]:
        reports_features_out_of_sample = topic_model_features(reports_out_of_sample, vectorizer, topic_model)

    # different covariance matrix predictions
    cov_sample = predict_cov_sample(returns_sample)
    cor_sample = predict_cov_sample(returns_sample, True)

    cov_upper = cov_sample[np.triu_indices(cov_sample.shape[0], k=1)]
    cor_upper = cor_sample[np.triu_indices(cor_sample.shape[0], k=1)]
    sample_mean_cov = cov_upper.mean()
    sample_mean_cor = cor_upper.mean()
    sample_mean_var = np.diagonal(cov_sample).mean()

    if predict_corr:
        cov_model = predict_correlation_matrix_model(lr, scaler, reports_features_out_of_sample, sample_mean_cor,
                                                     feature_wise, standardize_cov_matrix, cov_sample)
    else:
        cov_model = predict_covariance_matrix_model(lr, scaler, reports_features_out_of_sample, sample_mean_var,
                                                    sample_mean_cov, feature_wise, standardize_cov_matrix)

    cov_equal = np.full(cov_sample.shape, sample_mean_cov)
    np.fill_diagonal(cov_equal, sample_mean_var)

    cov_constant = constant_covariance_model(cov_sample, sample_mean_cor)

    LW = LedoitWolf()
    cov_lw = LW.fit(returns_sample).covariance_

    cov_combined = (cov_model + cov_lw) / 2

    # empirical variance for the out of sample time frame
    cov_true = compute_cov_matrix(returns_out_of_sample)

    frob_results_line = [sample_stop, np.linalg.norm(cov_true - cov_equal, ord="fro"),
                         np.linalg.norm(cov_true - cov_constant, ord="fro"),
                         np.linalg.norm(cov_true - cov_sample, ord="fro"), np.linalg.norm(cov_true - cov_lw, ord="fro"),
                         np.linalg.norm(cov_true - cov_model, ord="fro"),
                         np.linalg.norm(cov_true - cov_combined, ord="fro")]

    # evaluation of the predictions
    print("--------eval-------- ")

    print("frobenius norm for cov equal")
    print(frob_results_line[1])

    print("frobenius norm for cov constant")
    print(frob_results_line[2])

    print("frobenius norm for sample cov")
    print(frob_results_line[3])

    print("frobenius norm for ledoit wolf estimator")
    print(frob_results_line[4])

    print("frobenius norm using nlu model on risk reports")
    print(frob_results_line[5])

    print("frobenius norm of combined estimates of lw and model")
    print(frob_results_line[6])

    #writing results to csv file
    frob_rows.append(frob_results_line)

    # constructing portfolios based on predictions
    w_equal = optimal_portfolio_weights(cov_equal)
    w_constant = optimal_portfolio_weights(cov_constant)
    w_sample = optimal_portfolio_weights(cov_sample)
    w_lw = optimal_portfolio_weights(cov_lw)
    w_model = optimal_portfolio_weights(cov_model)
    w_combined = optimal_portfolio_weights(cov_combined)

    # calculating realized returns based on these portfolios
    print("-----compute returns----")

    print("equal cov portfolio")
    r_equal, r_equal_var = realized_portfolio_returns(returns_out_of_sample.values, w_equal, r_equal)
    print("constant cov portfolio")
    r_constant, r_constant_var = realized_portfolio_returns(returns_out_of_sample.values, w_constant, r_constant)
    print("sample cov portfolio")
    r_sample, r_sample_var = realized_portfolio_returns(returns_out_of_sample.values, w_sample, r_sample)
    print("ledoit wolf cov portfolio")
    r_lw, r_lw_var = realized_portfolio_returns(returns_out_of_sample.values, w_lw, r_lw)
    print("model cov portfolio")
    r_model, r_model_var = realized_portfolio_returns(returns_out_of_sample.values, w_model, r_model)
    print("combined cov portfolio")
    r_combined, r_combined_var = realized_portfolio_returns(returns_out_of_sample.values, w_combined, r_combined)

    var_line = [sample_stop, r_equal_var, r_constant_var, r_sample_var, r_lw_var, r_model_var, r_combined_var]

    var_rows.append(var_line)

#csv file writers to save results
with open("results/frob_" + trial_name + ".csv", "w", newline="") as file:
    frob_writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    frob_writer.writerows(frob_rows)
with open("results/var_" + trial_name + ".csv", "w", newline="") as file:
    var_writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    var_writer.writerows(var_rows)


# plotting the returns
x = range(len(r_equal))
plt.plot(x, r_equal, label="market")
# plt.plot(x, r_sample, label="sample")
plt.plot(x, r_lw, label="ledoit wolf")
# plt.plot(x, r_model, label="lda model")
plt.plot(x, r_combined, label="combined")

plt.legend()

plt.show()

print("coefficients of regression model")
print(lr.coef_)
