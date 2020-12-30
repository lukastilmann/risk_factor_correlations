import pandas as pd
import numpy as np

dir = r"C:\Users\Lukas Tilmann\Documents\uni\Semester 8\BA\stock_tickers\{}.csv"

file = "data/companies.csv"

df_companies = pd.read_csv(file, sep=";")

df_all = pd.DataFrame()

companies = df_companies.loc[:,["company", "ticker"]].dropna().drop_duplicates()

for i, row in companies.iterrows():
        df = pd.read_csv(dir.format(row["ticker"]), index_col="Date")
        df.index = pd.to_datetime(df.index)
        df["Move"] = df["Close"].pct_change()
        df_all[row["ticker"] + "_" + str(row["company"])] = df["Move"]

df_all.to_csv("data/stock_returns.csv")


