from edgar import Edgar, Company
import pandas as pd

companies = "data/companies.csv"
edgar = Edgar()
df_companies = pd.read_csv(companies, sep=";")

companies_unique = df_companies["company"].dropna().unique()


def is_item_1a_header(text):
    if "1A" in text:
        # print(text)
        text = text.replace("&nbsp", " ").strip().lower()
        if (len(text) < 30 and "item" in text and "1a" in text) and (
                len(text) < 15 or ("risk" in text or "factors" in text)):
            # print(text)
            return True

    return False


def is_item_1b_header(text):
    if "1B" in text:
        text = text.replace("&nbsp", " ").strip().lower()
        if (len(text) < 50 and "item" in text and "1b" in text) and (
                len(text) < 15 or ("unresolved" in text or "staff" in text or "comments" in text)):
            return True

    return False


def is_item_2_header(text):
    if len(text) < 150 and "2" in text:
        text = text.replace("&nbsp", " ").strip().lower()
        if (len(text) < 70 and "item" in text and "2" in text) and (
                len(text) < 15 or (any(word in text for word in ["unregistered", "equity", "securities", "proceeds"]))):
            return True

    return False

def print_filing(tuple):
  print(str(not tuple[3]) + " " + tuple[0] + " - " + tuple[1] + " -" + tuple[2][:250])

def parse_filing(company, doc_lxml, doc):
    text_elem = doc_lxml.xpath("//text()")

    fromhere = False
    current_text = ""
    risk_factors_text = []

    # print(text_elem)

    for text in text_elem:
        if fromhere:
            if is_item_1b_header(text) or is_item_2_header(text):
                fromhere = False
                if len(risk_factors_text) > 10:
                    print("got item 2 or 1b and has text!")
                    break
                else:
                    print("got next item but no text!")
            text = text.strip()
            if len(text) > 4 and not (
                    any(word in text.lower() for word in ["index", "contents", "item", "factor", "summary"]) and len(
                    text) < 75):
                if text[0] in ",;:":
                    current_text = current_text + text
                else:
                    current_text = current_text + " " + text
                if text[-1] in ".!?":
                    risk_factors_text.append(current_text + " ")
                    current_text = ""
        if is_item_1a_header(text):
            fromhere = True
            print("item 1a start")

    text_content = "\n".join(risk_factors_text)

    print(fromhere)
    print(len(text_content))
    print("--------------------------")

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

def pull_reports(cik, ticker, c_id, start):

    company = get_company_by_cik(cik)
    #yearly reports first
    '''
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
        parsed = parse_filing(company, filings_10k[0][i], filings_10k[1][i])
        dates.append(period)
        content.append(parsed[2])
        fromhere.append(parsed[3])
        has_content.append(len(parsed[2]) > 0)

    df_10k = pd.DataFrame({"date":dates, "content":content, "fromhere":fromhere, "has_content":has_content})

    print(df_10k["fromhere"])
    '''

    filings_10q = get_filings_by_company(company, "10-Q", 20)
    # check if enough
    #if pd.Timestamp(str(filings_10q[1][-1].content["Period of Report"])) > pd.Timestamp(year=2005, month=12, day=31):
    #    filings_10q = get_filings_by_company(company, "10-Q", 60)
    #if pd.Timestamp(str(filings_10q[1][-1].content["Period of Report"])) > pd.Timestamp(year=2005, month=12, day=31):
    #    filings_10q = get_filings_by_company(company, "10-Q", 96)

    dates = []
    content = []
    fromhere = []
    has_content = []

    for i in range(len(filings_10q[0])):
        period = pd.Timestamp(str(filings_10q[1][i].content["Period of Report"]))
        if period < start:
            break
        parsed = parse_filing(company, filings_10q[0][i], filings_10q[1][i])
        dates.append(period)
        content.append(parsed[2])
        fromhere.append(parsed[3])
        has_content.append(len(parsed[2]) > 0)

    df_10q = pd.DataFrame({"date": dates, "content": content, "fromhere": fromhere, "has_content": has_content})

    print(df_10q["fromhere"])




#for comapny_id in companies_unique:

df_mo = df_companies[df_companies["company"] == 71]

for i, row in df_mo.iterrows():
    pull_reports(row["cik"], row["ticker"], row["company"], pd.Timestamp(year=2005, month=12, day=31))








