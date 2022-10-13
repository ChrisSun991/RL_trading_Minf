import random
import pandas as pd
import datetime as datetime


random.seed(911)

# ---------------------------------------------------------
# experiment no |     train     |        test        |
# ---------------------------------------------------------
# experiment 1  | SP500_original |SP500_orignal_test |
# ---------------------------   ------------------------------
# experiment 2  |SP500_exclusives|     NYSETestSet    |
# ---------------------------------------------------------
# experiment 3  |  SP500_original|   NYSETestOrgSEt  |
# ---------------------------------------------------------

date_start = "1996-01-01"
date_end = "2020-10-01"

test_start = "2016-12-20"

# dataset_dict = {'SP500':"SP500Assets", 'DJIA':"DJIAAssets"}

DJIAAssets = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'PG', 'UNH', 'HD', 'DIS', 'VZ', 'CMCSA', 'ADBE', 'PFE',
        'BAC', 'INTC', 'T', 'WMT', 'MRK', 'KO', 'PEP', 'ABT', 'TMO', 'CSCO', 'CVX', 'NKE', 'XOM']

NYSEAssets = ['JPM', 'JNJ', 'WMT', 'BAC', 'PG', 'XOM', 'T', 'UNH', 'DIS',
        'VZ', 'HD', 'RDS-B', 'KO', 'MRK', 'CVX', 'WFC', 'PFE', 'TM', 'BA',
        'ORCL', 'NKE', 'MCD', 'MDT', 'ABT', 'BMY', 'UL', 'BHP', 'NVO',
        'AZN', 'TMO', 'RTX', 'BP', 'HON', 'LLY', 'UNP']

x = pd.read_csv('NYSE.csv')
x['IPO Date'] = pd.to_datetime(x['IPO Date'])
date_limit = datetime.datetime(2016, 12, 20)
a = x[x['IPO Date'] < date_limit]
a = a[~a.Symbol.isin(['COIN','GFS','CHK','BSY','ACI','RPRX'])]
a.loc[a["Symbol"] == "BRK.A", "Symbol"] = "BRK-A"
a.loc[a["Symbol"] == "BF.A", "Symbol"] = "BF-A"
a.loc[a["Symbol"] == "MKC.V", "Symbol"] = "MKC-V"
a.loc[a["Symbol"] == "STZ.B", "Symbol"] = "STZ-B"
a.loc[a["Symbol"] == "LEN.B", "Symbol"] = "LEN-B"
a.loc[a["Symbol"] == "HEI.A", "Symbol"] = "HEI-A"
a.loc[a["Symbol"] == "BIO.B", "Symbol"] = "BIO-B"
a.loc[a["Symbol"] == "WSO.B", "Symbol"] = "WSO-B"
a.loc[a["Symbol"] == "TAP.A", "Symbol"] = "TAP-A"

a_list = a['Symbol'].tolist()


SP500payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
SP500Assets = SP500payload[0]['Symbol'].values.tolist()

# SP500_exclusives = SP500payload[1]
# SP500_exclusives.columns = SP500_exclusives.columns.to_flat_index()
# SP500_exclusives.columns = ['date', 'added_ticker','added_security','removed_ticker','removed_security','reason']
# SP500_exclusives = list(set(SP500payload[1]['added_ticker'].tolist() + SP500payload[1]['removed_ticker'].tolist()))

SP500_exclusives = SP500payload[0]
SP500_exclusives = SP500_exclusives[['Symbol','Date first added']]
SP500_exclusives = SP500_exclusives.fillna('1957-03-04')
SP500_exclusives = SP500_exclusives.sort_values(by=['Date first added'])
SP500_exclusives['Date first added'][51] = '1957-03-04'
SP500_exclusives['Date first added'] = pd.to_datetime(SP500_exclusives['Date first added'])
SP500_exclusives = SP500_exclusives[(SP500_exclusives['Date first added']<"1995-01-01")]
SP500_exclusives = SP500_exclusives[~SP500_exclusives.Symbol.isin(['AEE','PARA','SRE','WRK','MET','FE','TPR'])]  #,'DRI','WAT','HIG','FCX'
SP500_exclusives = SP500_exclusives[~SP500_exclusives.Symbol.isin(['HWM','MAR'])]

SP500_exclusives.loc[SP500_exclusives["Symbol"] == "BF.B", "Symbol"] = "BF-B"
SP500_exclusives = SP500_exclusives['Symbol'].values.tolist()

# SP500_no_NYSE = [_ for _ in SP500Assets if _ not in NYSEAssets]


SP500_original = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'PG', 'UNH', 'HD', 'DIS', 'VZ', 'CMCSA', 'ADBE', 'PFE',
        'BAC', 'INTC', 'T', 'WMT', 'MRK', 'KO', 'PEP', 'ABT', 'TMO', 'CSCO', 'CVX', 'NKE', 'XOM']

SP500_orignal_test = [_ for _ in SP500Assets if _ not in SP500_original]
random.shuffle(SP500_orignal_test)
SP500_orignal_test =[_ for _ in SP500_orignal_test if _ in SP500_exclusives][0:25]
# ['C', 'PNW', 'TGT', 'CPB', 'OXY', 'NRG', 'AMZN', 'CMA', 'CSX', 'PPL', 'DVA', 'BA', 'WELL', 'BEN', 'COF', 'PSA', 'TROW',
#  'HAS', 'MCHP', 'AMAT', 'TAP', 'AXP', 'ORLY', 'A', 'IPG']

random.shuffle(SP500_exclusives)
SP500_exclusives = SP500_exclusives[0:150]

NYSETestOrgSEt = a_list[0:25]
NYSETestSet = a_list[0:150]



