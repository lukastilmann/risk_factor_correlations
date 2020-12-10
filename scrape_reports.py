from edgar import Edgar, Company
import pandas as pd
import os
import pickle

companies = "data/companies.csv"
edgar = Edgar()
df_companies = pd.read_csv(companies, sep=";")

companies_unique = df_companies["company"].dropna().unique()


def is_item_1a_header(text):
    if "1A" in text:
        # print(text)
        text = text.replace("&nbsp;", " ").strip().lower()
        if (len(text) < 31 and "tem" in text and "1a" in text) and (
                len(text) < 13 or ("risk" in text or "factors" in text)):
            # print(text)
            return True

    return False


def is_item_1b_header(text):
    if "1B" in text:
        text = text.replace("&nbsp;", " ").strip().lower()
        if (len(text) < 48 and "tem" in text and "1b" in text) and (
                len(text) < 13 or ("unresolved" in text or "staff" in text or "comments" in text)):
            return True

    return False


def is_10q_item_2_header(text):
    if len(text) < 150 and "2" in text:
        text = text.replace("&nbsp;", " ").strip().lower()
        if (len(text) < 85 and "tem" in text and "2" in text) and (
                len(text) < 12 or (any(word in text for word in ["unregistered", "equity", "securities", "proceeds"]))):
            return True

    return False

def is_10k_item_2_header(text):
    if len(text) < 150 and "2" in text:
        text = text.replace("&nbsp;", " ").strip().lower()
        if (len(text) < 30 and "tem" in text and "2" in text) and (
                len(text) < 12 or ("properties" in text)):
            return True

    return False

def is_item_6_header(text):
    if len(text) < 150 and "6" in text:
        text = text.replace("&nbsp;", " ").strip().lower()
        if (len(text) < 27 and "tem" in text and "6" in text) and (
                len(text) < 12 or ("exhibits" in text)):
            return True

    return False


def print_filing(tuple):
  print(str(not tuple[3]) + " " + tuple[0] + " - " + tuple[1] + " -" + tuple[2][:250])

def parse_10q_filing(company, doc_lxml, doc):
    text_elem = doc_lxml.xpath("//text()")

    fromhere = False
    current_text = ""
    risk_factors_text = []

    # print(text_elem)

    for text in text_elem:
        if fromhere:
            if is_item_1b_header(text) or is_10q_item_2_header(text) or is_item_6_header(text):
                fromhere = False
                if len(risk_factors_text) > 10:
                    #print("got item 2 or 1b and has text!")
                    break
                #else:
                #    print("got next item but no text!")
            text = text.replace("\n", " ").strip()
            if fromhere and len(text) > 4 and not (
                    any(word in text.lower() for word in ["index", "contents", "item", "factor", "summary", "form", "inc"]) and len(
                    text) < 40):
                if text[0] in ",;:":
                    current_text = current_text + text
                else:
                    current_text = current_text + " " + text
                if text[-1] in ".!?":
                    risk_factors_text.append(current_text + " ")
                    current_text = ""
        if is_item_1a_header(text):
            fromhere = True
            #print("item 1a start")

    text_content = "\n".join(risk_factors_text)


    return (company.name, doc.content["Period of Report"], text_content, fromhere)


def parse_10k_filing(company, doc_lxml, doc):
    text_elem = doc_lxml.xpath("//text()")

    fromhere = False
    current_text = ""
    risk_factors_text = []

    # print(text_elem)

    for text in text_elem:
        if fromhere:
            if is_item_1b_header(text) or is_10k_item_2_header(text):
                fromhere = False
                if len(risk_factors_text) > 5:
                    #print("got item 2 or 1b and has text!")
                    break
                #else:
                #    print("got next item but no text!")
            text = text.replace("\n", " ").strip()
            if len(text) > 4 and not (
                    any(word in text.lower() for word in ["index", "contents", "item", "factor", "summary"]) and len(
                    text) < 40):
                if text[0] in ",;:":
                    current_text = current_text + text
                else:
                    current_text = current_text + " " + text
                if text[-1] in ".!?":
                    risk_factors_text.append(current_text + " ")
                    current_text = ""
        if is_item_1a_header(text):
            fromhere = True
            #print("item 1a start")

    text_content = "\n".join(risk_factors_text)

    return (company.name, doc.content["Period of Report"], text_content, fromhere)

def get_company_by_cik(cik):
    cik_table = str(cik)
    length = len(cik_table)
    cik = "0" * (10 - length) + cik_table
    name = edgar.get_company_name_by_cik(cik)
    company = Company(name, cik)

    return company

def get_filings_by_company(company, type, n):
    tree = company.get_all_filings(filing_type=type)
    docs_lxml = Company.get_documents(tree, no_of_documents=n)
    docs_data = Company.get_documents(tree, no_of_documents=n, as_documents=True)

    return (docs_lxml, docs_data)

def pull_company_reports(cik, ticker, c_id, start):

    company = get_company_by_cik(cik)
    #yearly reports first
    filings_10k = get_filings_by_company(company, "10-K", 13)
    #check if enough
    if pd.Timestamp(str(filings_10k[1][-1].content["Period of Report"])) > pd.Timestamp(year=2005, month=12, day=31):
        filings_10k = get_filings_by_company(company, "10-K", 26)

    dates = []
    content = []
    fromhere = []
    has_content = []

    for i in range(len(filings_10k[0])):
        period = pd.Timestamp(str(filings_10k[1][i].content["Period of Report"]))
        if period < start:
            break
        parsed = parse_10k_filing(company, filings_10k[0][i], filings_10k[1][i])
        dates.append(period)
        content.append(parsed[2])
        fromhere.append(parsed[3])
        has_content.append(len(parsed[2]) > 0)

    df_10k = pd.DataFrame({"content":content, "fromhere":fromhere, "has_content":has_content}, index=dates)

    #print(df_10k)

    filings_10q = get_filings_by_company(company, "10-K", 20)
    # check if enough
    if pd.Timestamp(str(filings_10q[1][-1].content["Period of Report"])) > pd.Timestamp(year=2005, month=12, day=31):
        filings_10q = get_filings_by_company(company, "10-Q", 60)
    if pd.Timestamp(str(filings_10q[1][-1].content["Period of Report"])) > pd.Timestamp(year=2005, month=12, day=31):
        filings_10q = get_filings_by_company(company, "10-Q", 96)

    dates = []
    content = []
    fromhere = []
    has_content = []

    for i in range(len(filings_10q[0])):
        period = pd.Timestamp(str(filings_10q[1][i].content["Period of Report"]))
        if period < start:
            break
        parsed = parse_10q_filing(company, filings_10q[0][i], filings_10q[1][i])
        dates.append(period)
        content.append(parsed[2])
        fromhere.append(parsed[3])
        has_content.append(len(parsed[2]) > 0)

    df_10q = pd.DataFrame({"content": content, "fromhere": fromhere, "has_content": has_content}, index=dates)

    #print(df_10q["content"][:250] + " ---- " + df_10q["content"][-250:])
    return (df_10k, df_10q)

def scrape(location):

    if os.path.isfile(location):
        company_status_dict = pickle.load(open(location, "rb"))
    else:
        company_status_dict = {}
        for company_id in companies_unique:
            company_status_dict[company_id] = [False, False]
        pickle.dump(company_status_dict, open(location, "wb"))

    report_file_name = "data/{ticker}_{id}_{type}.csv"

    if os.path.isfile("list_issues.p"):
        list_issues = pickle.load(open("list_issues.p", "rb"))
    else:
        list_issues = []

    #i = 0

    for c_id in company_status_dict:
        #i += 1
        #if i > 2:
        #    break
        if not company_status_dict[c_id][0]:
            df_to_scrape = df_companies[df_companies["company"] == c_id]
            if len(df_to_scrape.index) > 1:
                list_10k = []
                list_10q = []
                for i, row in df_to_scrape.iterrows():
                    ticker = row["ticker"]
                    company_id = row["company"]
                    df_10k, df_10q = pull_company_reports(row["cik"], ticker, company_id, pd.Timestamp(year=2005, month=12, day=31))
                    list_10k.append(df_10k)
                    list_10q.append(df_10q)
                df_complete_10k = pd.concat(list_10k)
                df_complete_10q = pd.concat(list_10q)
            else:
                row = df_to_scrape.iloc[0]
                ticker = row["ticker"]
                company_id = row["company"]
                df_complete_10k, df_complete_10q = pull_company_reports(row["cik"], ticker, company_id, pd.Timestamp(year=2005, month=12, day=31))

            seems_correct = (not ( df_complete_10k["fromhere"].any() or df_complete_10q["fromhere"].any())) and df_complete_10k["has_content"].all()

            if not seems_correct:

                issues = [df_complete_10k["fromhere"].any(), df_complete_10q["fromhere"].any(), not df_complete_10k["has_content"].all()]
                list_issues.append((company_id, issues))
                pickle.dump(list_issues, open("list_issues.p", "wb"))


            df_complete_10q.to_csv(report_file_name.format(ticker = ticker, id=company_id, type="10-Q"))
            df_complete_10k.to_csv(report_file_name.format(ticker=ticker, id=company_id, type="10-K"))

            company_status_dict[c_id][0] = True
            company_status_dict[c_id][1] = seems_correct
            pickle.dump(company_status_dict, open(location, "wb"))
            print("scraped reports for {company}. status: {status}".format(company=ticker, status=seems_correct))

        else:
            ticker = df_companies[df_companies["company"] == c_id].iloc[0]["ticker"]
            print("the reports for {} have already been downloaded.".format(ticker))


    print("done!")



scrape("scrape_status_dict.p")