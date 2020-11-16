import pandas as pd
import numpy as np

stocks = ["AA", "AAPL", "ABBV", "ABT", "ACN", "XOM"]

dir = r"C:\Users\Lukas Tilmann\Documents\uni\Semester 8\BA\stock_tickers\{}.csv"


def create_data_frame(stocks):
    df_all = pd.DataFrame()
    for symbol in stocks:
        df = pd.read_csv(dir.format(symbol), index_col="Date")
        df.index = pd.to_datetime(df.index)
        df["Move"] = df["Close"].pct_change()
        df_all[symbol] = df["Move"]

    return df_all

def returns_for_period(df, start, stop):
    df_period = df[start:stop]
    df_period = df_period.dropna(1)

    return df_period

def stocks_in_period(df, start, stop):
    df_period = returns_for_period(df, start, stop)
    stocks = df_period.columns

    return stocks

def evaluate_prediction(prediction, true):
    error = prediction - true
    mean_squared_error = np.mean(np.square(error))

    return mean_squared_error

def create_mean_cov(cov_mat):
    without_diag = cov_mat[~np.eye(cov_mat.shape[0], dtype=bool)].reshape(cov_mat.shape[0], -1)
    mean_cov = np.mean(without_diag)
    mean_cov = np.full(cov_mat.shape, mean_cov)
    np.fill_diagonal(mean_cov, 1)

    return mean_cov

def run_predictions(df, start, horizon):
    max = df.index[-1] - horizon
    timestamps = []

    for year in range(start.year,max.year + 1):
        for month in range(1,11,3):
            timestamp = pd.Timestamp(year=year, month=month, day=1)
            if not timestamp > max:
                timestamps.append(timestamp)
            else:
                break

    print(timestamps)

df_all = create_data_frame(stocks)

run_predictions(df_all, pd.Timestamp(year=2005, month=1, day=1), pd.Timedelta(365, "days"))

'''
stocks_both = list(set(stocks_in_period(df_all, start_p1, stop_p1 )) |set(stocks_in_period(df_all, start_p2, stop_p2 )))


df_period_1 = returns_for_period(df_all, start_p1, stop_p1)[stocks_both]

df_period_2 = returns_for_period(df_all, start_p2, stop_p2)[stocks_both]


#cov_mat_period1 = np.corrcoef(df_period_1.values.transpose())

#cov_mat_period2 = np.corrcoef(df_period_2.values.transpose())

cov_mat_period1 = np.cov(df_period_1.values.transpose())

cov_mat_period2 = np.cov(df_period_2.values.transpose())

mean_pred = create_mean_cov(cov_mat_period1)

print(evaluate_prediction(cov_mat_period1, cov_mat_period2))

print(evaluate_prediction(mean_pred, cov_mat_period2))

#cov_mat = (np.corrcoef(returns_period.values.transpose()))

#print(cov_mat)
'''