from addpath import *
from os.path import join
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import *
import h5py
from variables import *


def getData(assets: [], indexName, start_date):
    if indexName == "SP_500_test":
        print(assets)
    d = {}
    for a in assets:
        d[a] = pdr.get_data_yahoo(a, start=start_date, end=date_end)
        ## for debug
        print("Number of Assets: {}".format(len(d)))
        print("Number of Steps: {}".format(len(d[assets[0]])))

    ## Transpose data
    for a_2 in assets:
        d[a_2] = d[a_2].interpolate(method='polynomial', order=2)
        d[a_2]['Daily Return'] = d[a_2]['Adj Close'].pct_change(1)
        d[a_2]['Log Return'] = np.log(d[a_2]["Adj Close"] / d[a_2]["Adj Close"].shift(1))
        d[a_2]['Code'] = a_2

    data = []
    for a_3 in assets:
        data.append(d[a_3].values.tolist())
    dates = d[assets[0]].index.tolist()
    dates = [obj.strftime("%d/%m/%Y") for obj in dates]

    results = []

    for i in range(len(data)):
        assets_temp = []
        for j in range(len(data[0])):
            try:
                assets_temp.append(data[i][j][:4])  ##
            except:
                print(data[i])

        results.append(assets_temp)
    results = np.array(results)
    abb = [abbr.encode() for abbr in assets]
    write_to_h5py(results, abb, dates, join(root_path, "datasets", "{}.h5").format(indexName))


if __name__ == '__main__':
    # ---------------------------------------------------------
    # experiment no |     train     |        test        |
    # ---------------------------------------------------------
    # experiment 1  | SP500_original |SP500_orignal_test |
    # ---------------------------------------------------------
    # experiment 2  |   SP500Assets |     NYSETestSet    |
    # ---------------------------------------------------------
    # experiment 3  |  SP500_original|   NYSETestOrgSEt  |
    # ---------------------------------------------------------

    # getData(SP500_original, 'SP_500_org',date_start)
    # getData(SP500_orignal_test, 'SP_500_test', test_start)
    getData(SP500_exclusives, 'SP_500',date_start)
    getData(NYSETestSet, 'NYSE_T1', test_start)
    getData(NYSETestOrgSEt, 'NYSE_T2', test_start)
# 6483