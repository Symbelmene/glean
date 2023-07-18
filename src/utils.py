import os
import math
import numpy as np
import pandas as pd
import plotly.express as px

import warnings
warnings.simplefilter("ignore")

from tqdm import tqdm
from tabulate import tabulate
from plotly import graph_objects as go
from multiprocessing import Pool
from datetime import datetime
from finclasses import addBaseIndicatorsToDf

import utils
import finclasses

from config import Config, Interval
cfg = Config()


def log(e, display=True):
    if display:
        print(e, flush=True)

    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)

    logPath = f'{cfg.LOG_DIR}/log.txt'
    if not os.path.exists(logPath):
        with open(logPath, 'w+') as wf:
            wf.write(f'{datetime.now()}: {e}\n')
    else:
        with open(logPath, 'a') as wf:
            wf.write(f'{datetime.now()}: {e}\n')


def loadRawStockData(ticker, interval):
    # Try to get the file and if it doesn't exist issue a warning
    try:
        if interval == Interval.DAY:
            df = pd.read_csv(f'{cfg.DATA_DIR_24_HOUR}/{ticker}.csv', index_col=0, parse_dates=['Date'])
        elif interval == Interval.FIVE_MINUTE:
            df = pd.read_csv(f'{cfg.DATA_DIR_5_MINUTE}/{ticker}.csv', index_col=0, parse_dates=['Datetime'])
        else:
            log(f'Unrecognised interval {interval}.')
            raise KeyError
    except FileNotFoundError as ex:
        print(ex)
        return None

    df = df[~df.index.duplicated(keep='first')]
    return df


def loadMultipleDFsAndMergeByColumnName(colName, sDate, eDate, interval, tickers):
    mult_df = pd.DataFrame()

    for x in tickers:
        df = loadRawStockData(x, interval)

        if not df.index.is_unique:
            df = df.loc[~df.index.duplicated(), :]

        if df.index[-1] < eDate or df.index[0] > sDate:
            log(f'WARNING: Ticker {x} is missing stock data in requested range!')
            continue

        df = df[(df.index >= sDate) & (df.index <= eDate)]
        df = addBaseIndicatorsToDf(df)

        mult_df[x] = df[colName]

    return mult_df


def getValidTickers(interval):
    if interval == Interval.DAY:
        inDir = cfg.DATA_DIR_24_HOUR
    elif interval == Interval.FIVE_MINUTE:
        inDir = cfg.DATA_DIR_5_MINUTE
    else:
        log('Interval not recognised in getValidTickers')
        raise KeyError

    return [ticker.split('.')[0] for ticker in os.listdir(inDir)]


def loadAllRawStockData(interval):
    tickers = getValidTickers(interval)
    return {ticker : loadRawStockData(ticker, interval) for ticker in tickers}


def findReturnPoints(wkDict):
    df = wkDict['df'][77:]
    dfRet = pd.DataFrame()
    for intvl in range(1, wkDict['maxRetInterval']):
        dfRet[f'interval_return_{intvl}'] = df['daily_return'].rolling(intvl).sum().shift(-1*intvl)
    df['labels'] = dfRet.apply(lambda row: any((val > wkDict['acceptableReturn']) & (val < wkDict['maxReturn']) for val in row), axis=1)
    return wkDict['ticker'], df


def findAllReturnPoints(dfDict, maxRetInterval, acceptableReturn, maxReturn=0.20):
    '''
    Finds all points in the dataset at which investing would produce at least
    <acceptableReturnPerc +float> within <maxRetTime datetime>
    '''

    workList = [{'ticker' : ticker,
                 'df' : df,
                 'maxRetInterval' : maxRetInterval,
                 'acceptableReturn' : acceptableReturn,
                 'maxReturn' : maxReturn} for ticker, df in dfDict.items()]

    with Pool(8) as p:
        dfList = list(tqdm(p.imap(findReturnPoints, workList), total=len(dfDict)))

    dfDict = {k: v for k, v in dfList}

    totalGoodPoints = sum([df['labels'].sum() for df in dfDict.values()])
    totalPoints     = sum([len(df) for df in dfDict.values()])
    print(f'Found {totalGoodPoints} / {totalPoints} valid data points for at least a {100*acceptableReturn}% return within {maxRetInterval} intervals')
    return dfDict


###################################################################
# PLOTTING
###################################################################

def plotBollinger(df, ticker=''):
    fig = go.Figure()

    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df['Low'],
                            close=df['Close'], name="Candlestick")

    upper_line = go.Scatter(x=df.index, y=df['upper_band'],
                            line=dict(color='rgba(250, 0, 0, 0.75)',
                                      width=1), name="Upper Band")

    mid_line = go.Scatter(x=df.index, y=df['middle_band'],
                          line=dict(color='rgba(0, 0, 250, 0.75)',
                                    width=0.7), name="Middle Band")

    lower_line = go.Scatter(x=df.index, y=df['lower_band'],
                            line=dict(color='rgba(0, 250, 0, 0.75)',
                                      width=1), name="Lower Band")

    fig.add_trace(candle)
    fig.add_trace(upper_line)
    fig.add_trace(mid_line)
    fig.add_trace(lower_line)

    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")

    fig.update_layout(title=ticker + " Bollinger Bands", showlegend=False)
    fig.update_layout(xaxis={'rangeslider': {'visible': False}})
    return fig


def plotCandle(df, ticker=''):
    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick")

    fig = go.Figure()
    fig.add_trace(candle)
    fig.update_layout(title=ticker + " Candle", showlegend=False)
    fig.update_layout(xaxis={'rangeslider': {'visible': False}})
    return fig


def plotIchimoku(df, ticker=''):
    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick")

    df1 = df.copy()
    fig = go.Figure()
    df['label'] = np.where(df['SpanA'] > df['SpanB'], 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()

    df = df.groupby('group')

    dfs = []
    for name, data in df:
        dfs.append(data)

    for df in dfs:
        fig.add_traces([go.Scatter(x=df.index, y=df.SpanA,
                                  line=dict(color='rgba(0,0,0,0)'))])

        fig.add_traces([go.Scatter(x=df.index, y=df.SpanB,
                                  line=dict(color='rgba(0,0,0,0)'),
                                  fill='tonexty',
                                  fillcolor=get_fill_color(df['label'].iloc[0]))])

    baseline = go.Scatter(x=df1.index, y=df1['Baseline'], line=dict(color='pink', width=2), name="Baseline")
    conversion = go.Scatter(x=df1.index, y=df1['Conversion'], line=dict(color='black', width=1), name="Conversion")
    lagging = go.Scatter(x=df1.index, y=df1['Lagging'], line=dict(color='purple', width=2), name="Lagging")
    span_a = go.Scatter(x=df1.index, y=df1['SpanA'], line=dict(color='green', width=2, dash='dot'), name="Span A")
    span_b = go.Scatter(x=df1.index, y=df1['SpanB'], line=dict(color='red', width=1, dash='dot'), name="Span B")

    fig.add_trace(candle)
    fig.add_trace(baseline)
    fig.add_trace(conversion)
    fig.add_trace(lagging)
    fig.add_trace(span_a)
    fig.add_trace(span_b)

    fig.update_layout(title=ticker + " Ichimoku", showlegend=False)
    fig.update_layout(xaxis={'rangeslider': {'visible': False}})
    return fig


def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'


###################################################################
# BANALYSIS
###################################################################

def mergeDfByColumn(col_name, sdate, edate, *tickers):
    # Will hold data for all dataframes with the same column name
    mult_df = pd.DataFrame()
    for ticker in tickers:
        df = utils.loadRawStockData(ticker, Interval.DAY)
        df = finclasses.addBaseIndicatorsToDf(df)
        mask = (df.index >= pd.to_datetime(sdate)) & (df.index <= pd.to_datetime(edate))
        mult_df[ticker] = df.loc[mask][col_name]
    return mult_df


def plotStockData(eDate, mult_df, sDate, portList):
    # Plot out prices for each stock
    print('Plotting stock prices in selected date...')
    fig = px.line(mult_df, x=mult_df.index, y=mult_df.columns)
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")
    fig.update_layout(height=800, width=1200, showlegend=True)
    fig.show()

    mult_cum_df = mergeDfByColumn('cum_return', sDate, eDate, *portList)
    # Plot out cumulative returns for each stock since beginning of 2017
    print('Plotting cumulative returns in selected date...')
    fig = px.line(mult_cum_df, x=mult_cum_df.index, y=mult_cum_df.columns)
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")
    fig.update_layout(height=800, width=1200, showlegend=True)
    fig.show()


def optimisePortfolio(portfolioList, sDate, eDate, plot=False, samples=10000):
    num_stocks = len(portfolioList)
    print('Loading and preprocessing data...')
    mult_df = mergeDfByColumn('Close', sDate, eDate, *portfolioList)

    if plot:
        plotStockData(eDate, mult_df, sDate, portfolioList)

    returns = np.log(mult_df / mult_df.shift(1))
    mean_ret = returns.mean() * 252  # 252 average trading days per year

    # Generate 11 random values that sum to 1
    print('Plotting Pareto front...')
    weights = np.random.random(num_stocks)
    weights /= np.sum(weights)
    # Provide return of portfolio using random weights over the whole dataset

    p_ret = []  # Returns list
    p_vol = []  # Volatility list
    p_SR  = []  # Sharpe Ratio list
    p_wt  = []  # Stock weights list
    retList = []
    for _ in tqdm(range(samples)):
        # Generate random weights
        p_weights = np.random.random(num_stocks)
        p_weights /= np.sum(p_weights)

        # Add return using those weights to list
        ret_1 = np.sum(p_weights * returns.mean()) * 252
        p_ret.append(ret_1)

        # Add volatility or standard deviation to list
        vol_1 = np.sqrt(np.dot(p_weights.T, np.dot(returns.cov() * 252, p_weights)))
        p_vol.append(vol_1)

        # Get Sharpe ratio
        SR_1 = (ret_1 - cfg.RISK_FREE_RATE) / vol_1
        p_SR.append(SR_1)

        # Store the weights for each portfolio
        p_wt.append(p_weights)

        retDict = {'return' : ret_1, 'volatility' : vol_1, 'sharpe_ratio': SR_1, 'weights': p_weights}
        retList.append(retDict)

    # Convert to Numpy arrays
    p_ret = np.array(p_ret)
    p_vol = np.array(p_vol)
    p_SR  = np.array(p_SR)
    p_wt  = np.array(p_wt)

    # Create a dataframe with returns and volatility
    ports = pd.DataFrame({'Return': p_ret, 'Volatility': p_vol})

    if plot:
        print('Plotting Pareto Front...')
        fig = px.scatter(ports, x=ports['Volatility'], y=ports['Return'])
        fig.update_layout(height=800, width=1200, showlegend=True)
        fig.show()

    # Return the index of the largest Sharpe Ratio
    SR_idx = np.argmax(p_SR)

    # Find the ideal portfolio weighting at that index
    idealPortfolioWeighting = [p_wt[SR_idx][i] * 100 for i in range(num_stocks)]

    return p_vol[SR_idx], p_ret[SR_idx], idealPortfolioWeighting


def getPortFolioShares(one_price, force_one, wts, prices):
    # Gets number of stocks to analyze
    num_stocks = len(wts)

    # Holds the number of shares for each
    shares = []

    # Holds Cost of shares for each
    cost_shares = []

    i = 0
    while i < num_stocks:
        # Get max amount to spend on stock
        max_price = one_price * wts[i]

        # Gets number of shares to buy and adds them to list
        num_shares = int(max_price / prices[i])

        # If the user wants to force buying one share do it
        if (force_one & (num_shares == 0)):
            num_shares = 1

        shares.append(num_shares)

        # Gets cost of those shares and appends to list
        cost = num_shares * prices[i]
        cost_shares.append(cost)
        i += 1

    return shares, cost_shares


def getPortfolioWeighting(share_cost):
    # Holds weights for stocks
    stock_wts = []
    # All values summed
    tot_val = sum(share_cost)
    print("Total Investment :", tot_val)

    for x in share_cost:
        stock_wts.append(x / tot_val)
    return stock_wts


def markowitzPortfolioOptimisation(portList, investValue):
    S_DATE = '2019-01-04'
    E_DATE = '2023-01-28'

    volatility, annualReturn, sharpeWeights = optimisePortfolio(portList, S_DATE, E_DATE, plot=False, samples=4000)
    sharpeWeights = [round(e, 3) for e in sharpeWeights]
    # Get all stock prices on the starting date
    port_df_start = mergeDfByColumn('Close', '2022-01-07', '2022-01-07', *portList)

    # Convert from dataframe to Python list
    port_prices = port_df_start.values.tolist()

    # Trick that converts a list of lists into a single list
    port_prices = np.array(sum(port_prices, []))
    portValues = investValue * np.array(sharpeWeights) / 100

    shareCounts = portValues / port_prices

    portDict = {'tickers'      : portList,
                'prices'       : port_prices,
                'distribution' : list(portValues),
                'sharpeweights': sharpeWeights,
                'shares'       : list(shareCounts)}

    portDf = pd.DataFrame(portDict)

    print(tabulate(portDf, headers='keys', tablefmt='psql'))
    print(f'Volatility    : {round(volatility, 3)}')
    print(f'Annual Return : {round(annualReturn, 3)}')


def calculateVolatility(col):
    return col.std() / math.sqrt(len(col))


def calculateReturn(col):
    return np.sum(col / col.shift(1))


def calculateSharpeRatio(col):
    return calculateReturn(col) / calculateVolatility(col)
