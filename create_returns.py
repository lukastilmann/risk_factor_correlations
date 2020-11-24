import pandas as pd
import numpy as np

dir = r"C:\Users\Lukas Tilmann\Documents\uni\Semester 8\BA\stock_tickers\{}.csv"

file = "data/companies.csv"

df_companies = pd.read_csv(file, sep=";")

df_all = pd.DataFrame()

tickers = df_companies["ticker"].dropna().unique()

for symbol in tickers:
        df = pd.read_csv(dir.format(symbol), index_col="Date")
        df.index = pd.to_datetime(df.index)
        df["Move"] = df["Close"].pct_change()
        df_all[symbol] = df["Move"]

df_all.to_csv("data/stock_retuns.csv")


