import pandas as pd
import datetime as dt

def get_prices(symbols, start_date, end_date):
    
    names=['date','ibovespa']
    dtypes = {'date':'str','ibovespa':'float'}
    parse_dates = ['date']

    result = pd.read_csv('./data/BVSP.csv', sep=',',header=0, names=names, usecols=[0,5], dtype=dtypes, parse_dates=parse_dates, index_col=0)
    
    for element in symbols:
        names=['date', element]
        dtypes = {'date':'str', element:'float'}
        parse_dates = ['date']

        df = pd.read_csv('./data/'+element+'.csv', sep=',',header=0, names=names, usecols=[0,5], dtype=dtypes, parse_dates=parse_dates, index_col=0)
        result = pd.concat([result, df], axis=1, join='inner')
    
    result.sort_index(inplace=True)
    result = result.loc[start_date:end_date]
    result.dropna(inplace=True)
    
    return result