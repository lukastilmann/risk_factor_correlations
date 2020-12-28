import pandas as pd

file = "data/companies.csv"

df_companies = pd.read_csv(file, sep=";")


def create_columns():
    companies = df_companies.loc[:,["company", "ticker"]].dropna().drop_duplicates()

    file_format = "data/{ticker}_{id}_{type}.csv"

    ind = 0

    df_reports = pd.DataFrame()


    for i, row in companies.iterrows():
        location_10k = file_format.format(ticker=row["ticker"], id=row["company"], type="10-K")
        location_10q = file_format.format(ticker=row["ticker"], id=row["company"], type="10-Q")

        df_10k = pd.read_csv(location_10k, index_col="Unnamed: 0").sort_index()
        df_10k.index = pd.to_datetime(df_10k.index)
        df_10q = pd.read_csv(location_10q, index_col="Unnamed: 0").sort_index()
        df_10q.index = pd.to_datetime(df_10q.index)

        reports = []
        dates = []

        for date in df_10k.index:
            quarter_end_date = date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd()

            report_10k = df_10k.loc[:quarter_end_date].iloc[-1]["content"]
            # print(report_10k)
            reports.append(report_10k)
            dates.append(quarter_end_date)

            last_date = df_10q.index[-1]

            for i in range(3):
                quarter_end_date = quarter_end_date + pd.Timedelta(days=1) + pd.tseries.offsets.QuarterEnd()
                if last_date < quarter_end_date:
                    break
                report = df_10q.loc[:quarter_end_date].iloc[-1]["content"]
                if not pd.isna(report):
                    content = report_10k + report
                    reports.append(content)
                    dates.append(quarter_end_date)

        df = pd.Series(reports, index=dates)

        print(df[df.index.duplicated()])

            #df_reports[row["ticker"] + "_" + str(row["company"])] = df

        ind += 1
        if ind >4:
            break

    #print(df_reports)



create_columns()