from edgar import Edgar, Company
import pandas as pd
import os
import pickle

companies = "data/companies.csv"
edgar = Edgar()
df_companies = pd.read_csv(companies, sep=";")

companies_unique = df_companies["company"].dropna().unique()

def is_item_header(nr, title, text, title_key_word):
    bare_len = len("item " + nr + title)

    text = text.replace("&nbsp;", "").replace(".", "").strip().lower()

    length = len(text)

    if length < len("item " + nr) + 4:
        if length < 4 + len(nr) and "m" in text and nr in text:
            return True
        if length < 5 + len(nr) and "em" in text and nr in text:
            return True
        if length < 6 + len(nr) and "tem" in text and nr in text:
            return True
        if "item" in text and nr in text:
            return True

    if length < bare_len + 5:
        if any(word in text for word in title_key_word):
            return True

    return False


def is_item_1a_header(text, prev):
    if len(text) == 1 and "A" in text and "1" in prev and len(prev.replace("&nbsp;", "").strip()) < 10:
        return True
    if "1A" in text and len(text) < 200:
        return is_item_header("1a", "risk factors", text, ["risk", "factor"])
    return False


def is_item_1b_header(text, prev):
    if len(text) == 1 and "B" in text and "1" in prev and len(prev.replace("&nbsp;", "").strip()) < 10 :
        return True
    if "1B" in text and len(text) < 250:
        return is_item_header("1b", "unresolved staff comments", text, ["unresolved", "staff", "comments"])
    return False



def is_10q_item_2_header(text, prev):
    if "2" in text and len(text) < 250:
        if len(text) < 3 and  "tem" in prev.lower() and len(prev.replace("&nbsp;", "").strip()) < 10:
            print("on prev: "+ prev)
            return True
        return is_item_header("2", "unregistered sales of equity sequrities and use of proceeds", text, ["unregistered", "sales", "equity", "securities", "proceeds"]) \
               or is_item_header("2", "unregistered sales of equity sequrities and issuer purchases of equity securities", text, ["unregistered", "sales", "equity", "securities", "purchases"])
    return False


def is_10k_item_2_header(text, prev):
    if "2" in text and len(text) < 200:
        if len(text) < 3 and  "tem" in prev.lower() and len(prev.replace("&nbsp;", "").strip()) < 10:
            print("on prev: " + prev)
            return True
        return is_item_header("2", "properties", text, ["properties"])
    return False

def is_item_6_header(text, prev):
    if "6" in text and len(text) < 250:
        if len(text) < 3 and  "tem" in prev.lower() and len(prev.replace("&nbsp;", "").strip()) < 10:
            return True
        return is_item_header("6", "exhibits", text, ["exhibits"]) or \
               is_item_header("6", "exhibits and reports on form 8-k", text, ["exhibits", "reports", "form"]) or \
               is_item_header("5", "exhibits", text, ["exhibits"])
    return False


def print_filing(tuple):
  print(str(not tuple[3]) + " " + tuple[0] + " - " + tuple[1] + " -" + tuple[2][:250])


def parse_10q_filing(company, doc_lxml, doc):
    text_elem = doc_lxml.xpath("//text()")

    fromhere = False
    current_text = ""
    risk_factors_text = []

    # print(text_elem)
    prev = ""

    for text in text_elem:
        if fromhere:
            if is_item_1b_header(text, prev) or is_10q_item_2_header(text, prev) or is_item_6_header(text, prev):
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
        if is_item_1a_header(text, prev):
            fromhere = True
        prev = text
            #print("item 1a start")

    text_content = "\n".join(risk_factors_text)


    return (company.name, doc.content["Period of Report"], text_content, fromhere)


def parse_10k_filing(company, doc_lxml, doc):
    text_elem = doc_lxml.xpath("//text()")

    fromhere = False
    current_text = ""
    risk_factors_text = []

    # print(text_elem)
    prev = ""

    for text in text_elem:
        if fromhere:
            if is_item_1b_header(text, prev) or is_10k_item_2_header(text, prev):
                print(text)
                fromhere = False
                if len(risk_factors_text) > 10:
                    #print("got item 2 or 1b and has text!")
                    break
                #else:
                #    print("got next item but no text!")
            text = text.replace("\n", " ").strip()
            if fromhere and len(text) > 3 and not (
                    any(word in text.lower() for word in ["index", "contents", "item", "factor", "summary", "form", "inc"]) and len(
                    text) < 40):
                if text[0] in ",;:":
                    current_text = current_text + text
                else:
                    current_text = current_text + " " + text
                if text[-1] in ".!?":
                    risk_factors_text.append(current_text + " ")
                    current_text = ""
        if is_item_1a_header(text, prev):
            fromhere = True
            print(text)
        prev = text
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
    if not isinstance(docs_lxml, list):
        docs_lxml = [docs_lxml]
        docs_data = [docs_data]

    return (docs_lxml, docs_data)

def pull_company_reports(cik, ticker, c_id, start):

    company = get_company_by_cik(cik)
    #yearly reports first
    filings_10k = get_filings_by_company(company, "10-K", 14)
    if pd.Timestamp(str(filings_10k[1][-1].content["Period of Report"])) < pd.Timestamp(year=2005, month=12, day=31):
        filings_10k = (filings_10k[0][:-1], filings_10k[1][:-1])
    #if len(filings_10k) < 2:
        #filings_10k = [filings_10k]
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

    df_10k = check_amends(df_10k)

    #print(df_10k)

    filings_10q = get_filings_by_company(company, "10-Q", 20)
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

def check_amends(df):
    df_nan = df[df["has_content"] == False]
    flags = []
    for i, row in df_nan.iterrows():
        df_that_date = df.loc[i]
        if df_that_date["has_content"].any():
            flags.append(True)
        else:
            flags.append(False)
    if all(flags):
        return df[df["has_content"]]
    else:
        return df

def scrape(location):

    if os.path.isfile(location):
        company_status_dict = pickle.load(open(location, "rb"))
    else:
        company_status_dict = {}
        #for company_id in companies_unique:
        #    company_status_dict[company_id] = [False, False]
        for issue in pickle.load(open("list_issues.p", "rb")):
            company_status_dict[issue[0]] = [False, False]
        pickle.dump(company_status_dict, open(location, "wb"))

    report_file_name = "data/{ticker}_{id}_{type}.csv"
    issues_file = "list_issues_4.p"

    if os.path.isfile(issues_file):
        list_issues = pickle.load(open(issues_file, "rb"))
    else:
        list_issues = []

    #i = 0


    for c_id in company_status_dict:
        #i += 1
        #if i > 2:
        #    break
        if not company_status_dict[c_id][0]:
        #if True:
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
                pickle.dump(list_issues, open(issues_file, "wb"))


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



scrape("scrape_status_dict_4.p")