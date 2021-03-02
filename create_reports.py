import pandas as pd

file = "data/companies.csv"

df_companies = pd.read_csv(file, sep=";")

def consolidate_reports(df):
    if len(df[df["has_content"]].index) == 1:
        return df[df["has_content"]]
    else:
        new_content = ""
        for i, row in df[df["has_content"]].iterrows():
            new_content += row["content"]
        new_df = pd.DataFrame({"content": [new_content]}, index=df.iloc[0].index)
        return new_df

def create_columns(with_duplicates=True):
    companies = df_companies.loc[:,["company", "ticker"]].dropna().drop_duplicates()

    file_format = "data/{ticker}_{id}_{type}.csv"

    ind = 0

    df_reports = pd.DataFrame()


    for i, row in companies.iterrows():
        location_10k = file_format.format(ticker=row["ticker"], id=row["company"], type="10-K")
        location_10q = file_format.format(ticker=row["ticker"], id=row["company"], type="10-Q")

        df_10k = pd.read_csv(location_10k).drop_duplicates()
        df_10k.index = df_10k.index = pd.to_datetime(df_10k["Unnamed: 0"])
        df_10k = df_10k.sort_index()
        df_10q = pd.read_csv(location_10q).drop_duplicates()
        df_10q.index = df_10q.index = pd.to_datetime(df_10q["Unnamed: 0"])
        df_10q = df_10q.sort_index()

        reports = []
        dates = []
        last_date = df_10q.index[-1]

        for date in df_10k.index.drop_duplicates():
            quarter_end_date = date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd()
            quarter_start_date = date - pd.offsets.QuarterBegin() + pd.Timedelta(days=1)

            df_date = df_10k.loc[quarter_start_date:quarter_end_date]

            length = len(df_date)

            if length == 1:
                report_10k = df_date.iloc[0]["content"]

            elif length > 1:
                df_date = consolidate_reports(df_date)
                report_10k = df_date.iloc[0]["content"]
            else:
                report_10k = pd.NA

            if not pd.isna(report_10k):
                reports.append(report_10k)
                dates.append(quarter_end_date)

                for i in range(3):

                    quarter_start_date = quarter_end_date + pd.Timedelta(days=1)
                    quarter_end_date = quarter_start_date + pd.tseries.offsets.QuarterEnd()
                    if last_date < quarter_end_date:
                        break
                    last_quarter_df = df_10q.loc[quarter_start_date:quarter_end_date]
                    length = len(last_quarter_df.index)
                    if length == 1:
                        report = last_quarter_df.iloc[0]["content"]

                    elif length > 1:
                        last_quarter_df = consolidate_reports(last_quarter_df)
                        report = last_quarter_df.iloc[0]["content"]

                    else:
                        report = pd.NA


                    if (not pd.isna(report)) and (len(report.split()) > 100):
                        content = report_10k + report
                        reports.append(content)
                        dates.append(quarter_end_date)
                        report_10k = content
                    else:
                        if with_duplicates:
                            reports.append(report_10k)
                            dates.append(quarter_end_date)


        if len(reports) > 0:
            df = pd.Series(reports, index=dates)
            if len(df[df.index.duplicated()]) == 0:
                df_reports[row["ticker"] + "_" + str(row["company"])] = df


    if with_duplicates:
        df_reports.to_csv("data/reports_with_duplicates_final.csv", index_label="date")
    else:
        df_reports.to_csv("data/reports_without_duplicates_final.csv", index_label="date")





    print(df_reports.head())


create_columns(True)