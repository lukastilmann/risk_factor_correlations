import pandas as pd
import numpy as np
import os

dir = r"C:\Users\Lukas Tilmann\Documents\uni\Semester 8\BA\stock_tickers"


for filename in os.listdir(dir):
    if filename.endswith("Historical Data.csv"):
        file = dir + "\\" + filename
        df = pd.read_csv(file)
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
        df = df.rename(columns={"Price":"Close"})
        ticker = filename.split(" ")[0]
        df.to_csv(dir + "\\" + ticker + ".csv", index=False)
