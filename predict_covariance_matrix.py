from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import os
import pickle
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import QuarterBegin, QuarterEnd, DateOffset
from pandas import Timedelta
import matplotlib.pyplot as plt
import csv


# train functions train one of the various feature embeddings

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
    vectorizer = TfidfVectorizer(stop_words="english", strip_accents="unicode", min_df=10, use_idf=True)
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


# projects text data into the space of a topic model
def topic_model_features(reports, vectorizer, model):
    if isinstance(reports, pd.DataFrame):
        reports = reports.iloc[0]

    vector_bow = vectorizer.transform(reports)

    vector_topics = model.transform(vector_bow)

    return vector_topics


# projects text data into tfidf space
def tfidf_features(reports, vectorizer):
    if isinstance(reports, pd.DataFrame):
        reports = reports.iloc[0]

    return vectorizer.transform(reports)


def industry_class_features(df_industries, index):
    ind = df_industries.loc[index]

    return ind


def get_returns_for_period(df, start, stop):
    df = df.loc[start:stop]
    # print(df)
    df = df.dropna(axis=1)

    return df


# computes covariance or correlation matrix and returns it as a pandas DataFrame with index and column labels
def compute_cov_matrix(returns, corr=False):
    if corr:
        matrix = np.corrcoef(returns.values.transpose())
    else:
        matrix = np.cov(returns.values.transpose())
    df = pd.DataFrame(matrix, index=returns.columns, columns=returns.columns)

    return df


# finds the intersection of the column values of a list of dataframes
def find_column_intersection(list_dfs):
    list_columns = []

    for df in list_dfs:
        list_columns.append(set(df.columns))

    in_all = set.intersection(*list_columns)

    list_dfs_new = []

    for df in list_dfs:
        list_dfs_new.append(df[in_all])

    return list_dfs_new


# computes sample covariance matrix of a sample as default, correlation matrix if corr=True
def predict_cov_sample(prev_sample, corr=False):
    return compute_cov_matrix(prev_sample, corr).values


# computes covariance matrix where every covariance value and each variance value is the mean of those values in
# previous sample
def predict_mean(prev_sample):
    cov_mat = prev_sample.values
    without_diag = cov_mat[~np.eye(cov_mat.shape[0], dtype=bool)].reshape(cov_mat.shape[0], -1)
    mean_cov = np.mean(without_diag)
    mean_cov = np.full(cov_mat.shape, mean_cov)
    mean_variance = np.mean(np.diagonal(cov_mat))
    np.fill_diagonal(mean_cov, mean_variance)

    return mean_cov


# computes portfolio weights with minimum variance given a covariance matrix
def optimal_portfolio_weights(sigma, objective="minimum variance", mu=None, r_f=None):
    size = sigma.shape[0]
    sigma = np.asmatrix(sigma)
    one = np.ones((size, 1))
    one = np.asmatrix(one)
    if objective == "sharpe":
        expected_excess_returns = mu - r_f
        expected_excess_returns = np.asmatrix(expected_excess_returns).T
        w_star = (np.linalg.inv(sigma) * expected_excess_returns) / (one.T * np.linalg.inv(sigma) * expected_excess_returns)
    elif objective == "minimum variance":
        w_star = (np.linalg.inv(sigma) * one) / (one.T * np.linalg.inv(sigma) * one)

    return w_star


# computes calue of a portfolio over time and also returns standard deviation of returns
def realized_portfolio_returns(returns, w_mat, returns_previous, start_date):
    #calculate returns of the whole portfolio based on portfolio weights
    w = np.asarray(w_mat).T
    portfolio_returns = returns * w
    whole_portfolio_returns = np.sum(portfolio_returns, axis=1)

    #calculate standard deviation and variance of portfolio returns
    return_var = whole_portfolio_returns.var()
    return_std = np.sqrt(return_var)

    # for sharpe ratio:
    portfolio_return = np.product(whole_portfolio_returns + 1) - 1
    # read earliest interest rate for time period and convert from percentage
    risk_free_return = df_risk_free_return.loc[start_date:].values[0, 0] * 0.01
    trading_days = len(whole_portfolio_returns)
    #annualizing the sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_return) / (return_std * np.sqrt(trading_days))

    print("sharpe ratio: " + str(sharpe_ratio))
    print("r std: " + str(return_std))
    print("-----")

    portfolio_returns = np.concatenate((returns_previous, whole_portfolio_returns))

    return portfolio_returns, return_std, sharpe_ratio


# computes value of a portfolio given returns
def realize_returns(start, portfolio_returns):
    port_returns = [start]
    for r in portfolio_returns:
        port_returns.append(port_returns[-1] * (1 + r))

    return port_returns


# computes exponential distance
def exp_dist(x_1, x_2):
    dist = np.absolute(x_1 - x_2)
    sim = np.exp(-1 * np.square(dist))

    return sim


# extracts covariance values and computes similarity values and returns both as vectors
def get_similarities_cov(mat, feature_data, sim_function, feature_wise, standardize):
    flat_upper = mat[np.triu_indices(mat.shape[0], k=1)]
    n = feature_data.shape[0]

    if standardize:
        cov_mean = flat_upper.mean()
        cov_mat = flat_upper - cov_mean
    else:
        cov_mat = flat_upper

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


# returns the covariance matrix estimated by a trained regression model
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


# returns the correlation matrix estimated by a trained regression model
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

    # transformation
    matrix = np.tanh(matrix)

    np.fill_diagonal(matrix, 1)
    diag = np.diag(cov_mat)
    matrix = corr_matrix_to_cov_matrix(matrix, diag)

    return matrix


# converts a correlation matrix to a covariance matrix by using variance estimates (diag).
def corr_matrix_to_cov_matrix(cor_matrix, diag):
    v = np.asmatrix(diag)
    v_matrix = np.sqrt(np.matmul(v.T, v))
    cov_est = np.multiply(cor_matrix, v_matrix)

    return cov_est


# implements constant covariance model - assumes constant correlation
def constant_covariance_model(sample_cov, mean_corr):
    diag = np.diag(sample_cov)
    cor_matrix = np.full(sample_cov.shape, mean_corr)
    np.fill_diagonal(cor_matrix, 1)
    cov_est = corr_matrix_to_cov_matrix(cor_matrix, diag)

    return cov_est


# implements the model from the paper Covariance Regression Analysis by Zou et al.
def predict_cov_window_model(prev_cov, feature_data_prev, feature_data_next):
    sim_measure = lambda x_1, x_2: cosine_similarity(x_1.reshape(1, -1), x_2.reshape(1, -1))

    matrix_prev = pairwise_distances(feature_data_prev, metric=sim_measure)
    np.fill_diagonal(matrix_prev, 0)
    intercept_matrix = np.zeros(matrix_prev.shape)
    np.fill_diagonal(intercept_matrix, 1)

    train_x = np.concatenate([intercept_matrix.reshape((1, -1)), matrix_prev.reshape(1, -1)], axis=1)
    train_y = prev_cov.reshape(1, -1)

    lin_reg = LinearRegression(fit_intercept=False).fit(train_x, train_y)

    matrix_next = pairwise_distances(feature_data_next, metric=sim_measure)
    np.fill_diagonal(matrix_next, 0)
    intercept_matrix = np.zeros(matrix_next.shape)
    np.fill_diagonal(intercept_matrix, 1)

    next_x = np.concatenate([intercept_matrix.reshape((1, -1)), matrix_prev.reshape(1, -1)], axis=1)

    pred_y = lin_reg.predict(next_x)
    matrix_pred = np.reshape(pred_y, matrix_next.shape)

    return matrix_pred


#predicting correlation matrix based on industry classification and risk report data
def predict_correlation_matrix_both(model, scaler, feature_data_reports, feature_data_ind, mean_cor, add_mean, cov_mat):

    # calculating the two similarity matrices
    sim_matrix_reports = cosine_similarity(feature_data_reports)
    sim_matrix_industry = cosine_similarity(feature_data_ind)

    # predicting covariance based on both data types
    model_input = np.concatenate([np.reshape(sim_matrix_reports, (-1,1)), np.reshape(sim_matrix_industry, (-1,1))], axis=1)
    model_input = scaler.transform(model_input)
    sim_predictions = model.predict(model_input)
    if add_mean:
        sim_predictions = sim_predictions + mean_cor

    # reshape into matrix form
    matrix = np.reshape(sim_predictions, sim_matrix_reports.shape)

    # transformation
    matrix = np.tanh(matrix)

    # place 1 on diagonal and turn into covariance matrix
    np.fill_diagonal(matrix, 1)
    diag = np.diag(cov_mat)
    matrix = corr_matrix_to_cov_matrix(matrix, diag)

    return matrix

# parameters:
# how many quarters are in one sub-period
time_horizon_quarters = 1
# frequency of returns
frequency = "daily"
# can be "eval" for evaluating hyperparameters on the 2017-2018 sample or test for testing on 2019-2020 sample
mode = "eval"
#objective, can be "minimum variance" or "sharpe"
objective = "minimum variance"
# can be "reports", "industry" or "both"
which_data = "both"
# if "window", the model is trained on the preceding sample only
# if "whole", a model trained on the whole period before the test/eval sample is used
model_train_sample = "whole"
# the feature embedding that is used. Options are "lda", "tfidf" and "lsa"
model = "lda"
# if using tfidf-space as embedding, this specifies if inverse document frrequency-weighting is applied
idf = True
# if true, feature-wise similarity measure is used
feature_wise = False
# when using topic model (lsa or lda), this specifies the number of topics
n_dims = 5
# specifies if the regression model is fit with intercept
with_intercept = False
# specifies if cov-matrix is standardized by subtracting the mean covariance
standardize_cov_matrix = True
# if true, the model predicts correlation, not covariance. Only works with model trained on whole sample, not window
predict_corr = True
# the weight applied to the estimation generated from model when using the ensemble of model and lw-estimator
ensemble_weight = 0.2


# creating a name for the configuration under which to save results
dim = str(n_dims) if model in ["lda", "svd"] else ""
hor = str(time_horizon_quarters) + "Q"
model_save = "tfidf" if model == "tfidf" and idf else model
target = "cor" if predict_corr else "cov"
trial_name = "{model}{dim}_{target}_{horizon}_{freq}_{sample}_{ensemble_weight}_{mode}_{objective}"
trial_name = trial_name.format(model=model_save, dim=dim, target=target, horizon=hor, freq=frequency,
                               sample=model_train_sample, ensemble_weight=str(ensemble_weight),
                               mode=mode, objective=objective)

# loading data
df_reports = pd.read_csv("data/reports_with_duplicates_final.csv", dtype="string", index_col="date")
df_reports.index = pd.to_datetime(df_reports.index)
if frequency == "daily":
    df_returns = pd.read_csv("data/stock_returns.csv", index_col="Date")
if frequency == "weekly":
    df_returns = pd.read_csv("data/stock_returns_weekly.csv", index_col="Date")
df_returns.index = pd.to_datetime(df_returns.index)

# loading industry classifications and one hot encoding them
df_industry_classification = pd.read_csv("data/GISC.csv", sep=";",
                                         usecols=["company", "ticker", "GISC", "TRBC BusinessSector"])
df_industry_classification["indx"] = df_industry_classification["ticker"] + "_" + df_industry_classification[
    "company"].astype("str")
df_industry_classification.index = df_industry_classification["indx"]
df_industry_classification = df_industry_classification.drop(
    columns=["company", "ticker", "indx", "TRBC BusinessSector"])
oh_encoder = OneHotEncoder()
df_industry_classification = pd.get_dummies(df_industry_classification, columns=["GISC"])

# loading data for risk free rates of return (three month treasury bills)
df_risk_free_return = pd.read_csv("data/T_Bills_3m.csv", sep=";", decimal=",", index_col="Name", parse_dates=["Name"],
                                  dayfirst=True)

# defining train test split
train_first = datetime(year=2005, month=12, day=31)
if mode == "eval":
    train_last = datetime(year=2016, month=9, day=30)
if mode == "test":
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
mean_sims = []

if model_train_sample == "whole":
    for date in train_range:

        returns_stop = date + QuarterEnd(startingMonth=3, n=time_horizon_quarters)

        print("training for period:")
        print(date + pd.DateOffset(days=1))
        print(returns_stop)

        # loading reports and returns for period, finding the companies in which both datapoints exist
        reports = get_reports_for_date(df_reports, date)
        returns = get_returns_for_period(df_returns, date + pd.DateOffset(days=1), returns_stop)
        returns, reports = find_column_intersection([returns, reports])

        # covariance and correlation matrix for period
        cov = predict_cov_sample(returns)
        cor = predict_cov_sample(returns, True)

        # feature engineering
        if which_data == "reports" or "both":
            if model == "tfidf":
                feature_data_reports = tfidf_features(reports, vectorizer)
            if model in ["svd", "lda"]:
                feature_data_reports = topic_model_features(reports, vectorizer, topic_model)

        if which_data == "industry" or "both":
            feature_data_industry = industry_class_features(df_industry_classification, reports.columns)

        if predict_corr:
            est_target = cor
        else:
            est_target = cov

        # computing similarity matrix and getting the upper diagonal of covariance- and similarity-matrix
        # in a 1-dimensional vector

        # for risk reports
        if which_data == "reports" or "both":
            if feature_wise:
                sim_reports, est_target_upper_dig = get_similarities_cov(est_target, feature_data_reports, exp_dist, feature_wise=True,
                                                             standardize=standardize_cov_matrix)
            else:
                sim_reports, est_target_upper_dig = get_similarities_cov(est_target, feature_data_reports, cosine_similarity,
                                                             feature_wise=False, standardize=standardize_cov_matrix)

        # for industry classifications
        if which_data == "industry" or "both":
            sim_industry, est_target_upper_dig = get_similarities_cov(est_target, feature_data_industry, cosine_similarity, feature_wise=False,
                                                   standardize=standardize_cov_matrix)

        #adding to training data:
        if which_data == "reports":
            train_x.append(sim_reports)
        if which_data == "industry":
            train_x.append(sim_industry)
        if which_data == "both":
            row = np.stack((sim_reports, sim_industry), axis=1)
            train_x.append(row)

        train_y.append(est_target_upper_dig)

        # printing mean covariance of period
        cor_upper = cor[np.triu_indices(cor.shape[0], k=1)]
        print(cor_upper.mean())

    # creating the training data
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    # transformation of correlations, if those are estimation target
    if predict_corr:
        train_y = np.arctanh(train_y)

    # standardizing the data
    scaler = StandardScaler()
    if not feature_wise:
        if which_data == "both":
            train_x = train_x.reshape(-1,2)
        else:
            train_x = train_x.reshape(-1, 1)
    train_x = scaler.fit_transform(train_x)

    # regression model for prediction
    lr = LinearRegression(fit_intercept=with_intercept)
    lr.fit(train_x, train_y)

    print("model coefficients: ")
    print(lr.coef_)

# mean variance in the sample, as model doesn't predict variance
# sample_mean_var = np.mean(mean_covs)

# the following is to test to trained model
total_quarters = 8
test_intervals = int(total_quarters / time_horizon_quarters)

# initializing empty numpy arrays to save returns to
r_equal = np.array([])
r_constant = np.array([])
r_sample = np.array([])
r_lw = np.array([])
r_model = np.array([])
r_combined = np.array([])

# this is for saving the results
df_columns = ["equal", "constant", "sample", "lw", "model", "combined"]
df_index = []
frob_rows = []
var_rows = []
sharpe_rows = []

# testing the model
for i in range(test_intervals):

    if mode == "eval":
        y = 2016
    if mode == "test":
        y = 2018
    sample_start = datetime(year=y, month=1, day=1) + QuarterBegin(startingMonth=1, n=i * time_horizon_quarters)
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
    if which_data == "reports" or "both":
        if model == "tfidf":
            features_out_of_sample_reports = tfidf_features(reports_out_of_sample, vectorizer)
        if model in ["svd", "lda"]:
            features_out_of_sample_reports = topic_model_features(reports_out_of_sample, vectorizer, topic_model)
    if which_data == "industry" or "both":
        features_out_of_sample_industry = industry_class_features(df_industry_classification, reports_out_of_sample.columns)

    if which_data == "reports":
        features_out_of_sample = features_out_of_sample_reports
    if which_data == "industry":
        features_out_of_sample = features_out_of_sample_industry

    # different covariance matrix predictions
    cov_sample = predict_cov_sample(returns_sample)
    cor_sample = predict_cov_sample(returns_sample, True)

    cov_upper = cov_sample[np.triu_indices(cov_sample.shape[0], k=1)]
    cor_upper = cor_sample[np.triu_indices(cor_sample.shape[0], k=1)]
    sample_mean_cov = cov_upper.mean()
    sample_mean_cor = cor_upper.mean()
    sample_mean_var = np.diagonal(cov_sample).mean()

    if model_train_sample == "whole":
        if which_data == "both":
            LW = LedoitWolf()
            cov_lw = LW.fit(returns_sample).covariance_
            cov_model = predict_correlation_matrix_both(lr, scaler, features_out_of_sample_reports,
                                                        features_out_of_sample_industry, sample_mean_cor,
                                                        standardize_cov_matrix, cov_lw)
        else:
            if predict_corr:
                LW = LedoitWolf()
                cov_lw = LW.fit(returns_sample).covariance_
                cov_model = predict_correlation_matrix_model(lr, scaler, features_out_of_sample, sample_mean_cor,
                                                         feature_wise, standardize_cov_matrix, cov_lw)
            else:
                cov_model = predict_covariance_matrix_model(lr, scaler, features_out_of_sample, sample_mean_var,
                                                        sample_mean_cov, feature_wise, standardize_cov_matrix)

    if model_train_sample == "window":
        # feature engineer for the sample window
        if model == "tfidf":
            reports_features_sample = tfidf_features(reports_sample, vectorizer)
        if model in ["svd", "lda"]:
            reports_features_sample = topic_model_features(reports_sample, vectorizer, topic_model)

        cov_model = predict_cov_window_model(cov_sample, reports_features_sample, features_out_of_sample)

    # covariance estimate on assumption that all covariance values are equal
    cov_equal = np.full(cov_sample.shape, sample_mean_cov)
    np.fill_diagonal(cov_equal, sample_mean_var)

    # estimates from constant covariance model (
    cov_constant = constant_covariance_model(cov_sample, sample_mean_cor)

    # estimates from ledoit-wolf estimator
    LW = LedoitWolf()
    cov_lw = LW.fit(returns_sample).covariance_

    # weighted average of the estimation from the model and the sample covariance matrix
    cov_combined = ensemble_weight * cov_model + (1 - ensemble_weight) * cov_lw

    # empirical variance for the out of sample time frame
    cov_true = compute_cov_matrix(returns_out_of_sample)

    # the frobenius error norms for different estimates
    frob_results_line = [np.linalg.norm(cov_true - cov_equal, ord="fro"),
                         np.linalg.norm(cov_true - cov_constant, ord="fro"),
                         np.linalg.norm(cov_true - cov_sample, ord="fro"), np.linalg.norm(cov_true - cov_lw, ord="fro"),
                         np.linalg.norm(cov_true - cov_model, ord="fro"),
                         np.linalg.norm(cov_true - cov_combined, ord="fro")]

    df_index.append(sample_stop)
    frob_rows.append(frob_results_line)

    # evaluation of the predictions
    print("--------eval-------- ")

    print("frobenius norm for cov equal")
    print(frob_results_line[0])

    print("frobenius norm for cov constant")
    print(frob_results_line[1])

    print("frobenius norm for sample cov")
    print(frob_results_line[2])

    print("frobenius norm for ledoit wolf estimator")
    print(frob_results_line[3])

    print("frobenius norm using nlu model on risk reports")
    print(frob_results_line[4])

    print("frobenius norm of combined estimates of lw and model")
    print(frob_results_line[5])

    # constructing portfolios based on predictions
    # either the minimum variance portfolio or the portfolio on the efficient frontier
    # which maximizes the sharpe ratio
    if objective == "sharpe":
        r_f = ((1 + df_risk_free_return.loc[out_of_sample_start:].values[0, 0] * 0.01) ** (4/365))  - 1
        #equally weighted here
        df_returns_with_market = returns_sample.copy()
        df_returns_with_market["market"] = df_returns_with_market.mean(axis=1)
        cov_with_market = predict_cov_sample(df_returns_with_market)
        cov_m = cov_with_market[-1]
        beta = cov_m / np.diagonal(cov_with_market)
        mu = (r_f + beta * (0.0005 - r_f))[:-1]
    else:
        r_f = None
        mu = None

    w_equal = optimal_portfolio_weights(cov_equal, objective=objective, r_f=r_f, mu=mu)
    w_constant = optimal_portfolio_weights(cov_constant, objective=objective, r_f=r_f, mu=mu)
    w_sample = optimal_portfolio_weights(cov_sample, objective=objective, r_f=r_f, mu=mu)
    w_lw = optimal_portfolio_weights(cov_lw, objective=objective, r_f=r_f, mu=mu)
    w_model = optimal_portfolio_weights(cov_model, objective=objective, r_f=r_f, mu=mu)
    w_combined = optimal_portfolio_weights(cov_combined, objective=objective, r_f=r_f, mu=mu)

    # calculating realized returns based on these portfolios
    print("-----compute returns----")

    print("equal cov portfolio")
    r_equal, r_equal_var, sharpe_equal = realized_portfolio_returns(returns_out_of_sample.values, w_equal, r_equal,
                                                      out_of_sample_start)
    print("constant cov portfolio")
    r_constant, r_constant_var, sharpe_constant = realized_portfolio_returns(returns_out_of_sample.values, w_constant, r_constant,
                                                            out_of_sample_start)
    print("sample cov portfolio")
    r_sample, r_sample_var, sharpe_sample = realized_portfolio_returns(returns_out_of_sample.values, w_sample, r_sample,
                                                        out_of_sample_start)
    print("ledoit wolf cov portfolio")
    r_lw, r_lw_var, sharpe_lw = realized_portfolio_returns(returns_out_of_sample.values, w_lw, r_lw, out_of_sample_start)
    print("model cov portfolio")
    r_model, r_model_var, sharpe_model = realized_portfolio_returns(returns_out_of_sample.values, w_model, r_model,
                                                      out_of_sample_start)
    print("combined cov portfolio")
    r_combined, r_combined_var, sharpe_combined = realized_portfolio_returns(returns_out_of_sample.values, w_combined, r_combined,
                                                            out_of_sample_start)

    var_line = [r_equal_var, r_constant_var, r_sample_var, r_lw_var, r_model_var, r_combined_var]
    var_rows.append(var_line)

    sharpe_line = [sharpe_equal, sharpe_constant, sharpe_sample, sharpe_lw, sharpe_model, sharpe_combined]
    sharpe_rows.append(sharpe_line)

# code below creates the csv files in which results are saved
df_frob = pd.DataFrame(frob_rows, columns=df_columns, index=df_index)
df_frob.loc["all"] = df_frob.mean(axis=0)
df_frob["impr_model"] = (df_frob["model"] / df_frob["equal"]) - 1
df_frob["impr_comb"] = (df_frob["combined"] / df_frob["equal"]) - 1

print(df_frob)

std_whole = [r_equal.std(), r_constant.std(), r_sample.std(), r_lw.std(), r_model.std(), r_combined.std()]
df_var = pd.DataFrame(var_rows, columns=df_columns, index=df_index)
df_var.loc["mean"] = df_var.mean(axis=0)
df_var.loc["whole"] = std_whole
df_var["impr_model"] = (df_var["model"] / df_var["equal"]) - 1
df_var["impr_comb"] = (df_var["combined"] / df_var["equal"]) - 1

print(df_var)

df_sharpe = pd.DataFrame(sharpe_rows, columns=df_columns, index=df_index)
df_sharpe.loc["mean"] = df_sharpe.mean(axis=0)
df_sharpe["impr_model"] = (df_sharpe["model"] / df_sharpe["equal"]) - 1
df_sharpe["impr_comb"] = (df_sharpe["combined"] / df_sharpe["equal"]) - 1

print(df_sharpe)

obj = "variance of returns" if objective == "minimum variance" else "sharpe ratio"
print("improvement in {} through ensemble at weight of ".format(obj) + str(ensemble_weight) + ":")
if objective == "minimum variance":
    print(df_var.loc["mean", "impr_comb"])
if objective == "sharpe":
    print(df_sharpe.loc["mean", "impr_comb"])

port_r_equal = realize_returns(100, r_equal)
port_r_model = realize_returns(100, r_combined)
port_r_lw = realize_returns(100, r_lw)
port_r_sample = realize_returns(100, r_sample)

df_frob.to_csv("results/frob_" + trial_name + ".csv", sep=";")
df_var.to_csv("results/std_" + trial_name + ".csv", sep=";")
df_sharpe.to_csv("results/sharpe_" + trial_name + ".csv", sep=";")

# plotting the returns
x = range(len(port_r_equal))
plt.plot(x, port_r_lw, label="ledoit wolf")
plt.plot(x, port_r_model, label="model")
plt.plot(x, port_r_sample, label="sample")

plt.legend()
plt.xlabel("trading days")
plt.ylabel("portfolio value")

# plt.savefig("realized.png")
plt.show()
