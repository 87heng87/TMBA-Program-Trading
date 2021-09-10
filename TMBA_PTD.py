import io
import pandas as pd
from datetime import datetime
import numpy as np
import requests
import matplotlib.pyplot as plt


plt.style.use('ggplot')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


'''Yahoo Finance 爬蟲'''
def YahooData(ticker, start, end):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
    }

    url = "https://query1.finance.yahoo.com/v7/finance/download/" + str(ticker)
    x = int(datetime.strptime(start, '%Y-%m-%d').strftime("%s"))
    y = int(datetime.strptime(end, '%Y-%m-%d').strftime("%s"))
    url += "?period1=" + str(x) + "&period2=" + str(y) + "&interval=1d&events=history&includeAdjustedClose=true"

    r = requests.get(url, headers=headers)
    pad = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)

    return pad


'''布林通道策略'''
def BBands(x: pd.DataFrame, model, n, zu, zd):
    if model == 'o':
        ma = x['Close'].rolling(window=n).mean()
        sd = x['Close'].rolling(window=n).std()
        up = ma + zu * sd
        lo = ma - zd * sd
        bands = pd.concat([ma, up, lo], axis=1)
        bands.columns = ['Mid', 'Up', 'Lo']
        return bands
    elif model == 'm':
        a = 2 / (n + 1)
        Mt = x['Close'].ewm(span=n).mean()
        Ut = Mt.ewm(span=n).mean()
        Dt = ((2 - a) * Mt - Ut) / (1 - a)
        mt = abs(x['Close'] - Dt).ewm(span=n).mean()
        ut = mt.ewm(span=n).mean()
        dt = ((2 - a) * mt - ut) / (1 - a)
        bu = Dt + zu * dt
        bl = Dt - zd * dt
        Dt[0:19] = np.nan
        bu[0:19] = np.nan
        bl[0:19] = np.nan
        bands = pd.concat([Dt, bu, bl], axis=1)
        bands.columns = ['Mid', 'Up', 'Lo']
        return bands
    else:
        print('Method should be o or m')


'''信號產出函式，收盤價向上突破布林通道上限時開始隔日多單進場（空單出場），收盤價向下突破布林通道下限時隔日空單進場（多單出場）'''
def Signal(data: pd.DataFrame):
    long_signal = pd.Series(np.where(((data['Close'].shift(1) > data['Up'].shift(1)) & (data['Close'].shift(2) < data['Up'].shift(2)) &
                                      (data['Volume'].shift(1) > data['Volume'].shift(1).rolling(20).mean())), 1, np.nan), index=data.index)
    short_signal = pd.Series(np.where(((data['Close'].shift(1) < data['Lo'].shift(1)) & (data['Close'].shift(2) > data['Lo'].shift(2)) &
                                      (data['Volume'].shift(1) > data['Volume'].shift(1).rolling(20).mean())), -1, np.nan), index=data.index)
    signals = pd.concat([long_signal, short_signal], axis=1)
    signals.columns = ['long', 'short']
    signals['all'] = signals['short'].copy()
    signals.loc[pd.isnull(signals['all']), 'all'] = signals['long']
    signals['all'] = signals['all'].ffill().fillna(0)
    signals['long'] = pd.Series(np.where(signals['all'] == 1, 1, 0), index=data.index)
    signals['short'] = pd.Series(np.where(signals['all'] == -1, -1, 0), index=data.index)
    return signals


'''報酬率採用每日報酬率累積總和之方式計算'''
def Returns(data: pd.DataFrame, signal: pd.DataFrame):
    discount = 0.6
    fee = 1.425/1000 * discount
    ret_df = pd.concat([data['Close'].pct_change(), signal['all'], signal['long'], signal['short']], axis=1)
    ret_df.columns = ['d_ret', 'position_a', 'position_l', 'position_s']
    ret_df['return_a'] = ret_df['d_ret'] * ret_df['position_a']
    ret_df['return_a'] = np.where((ret_df['position_a'] != ret_df['position_a'].shift(1)), ret_df['return_a'] - fee, ret_df['return_a'])
    ret_df['return_l'] = ret_df['d_ret'] * ret_df['position_l']
    ret_df['return_l'] = np.where((ret_df['position_l'] != ret_df['position_l'].shift(1)), ret_df['return_l'] - fee, ret_df['return_l'])
    ret_df['return_s'] = ret_df['d_ret'] * ret_df['position_s']
    ret_df['return_s'] = np.where((ret_df['position_s'] != ret_df['position_s'].shift(1)), ret_df['return_s'] - fee, ret_df['return_s'])
    return ret_df


'''每年的報酬率'''
def Yearly_return(data: pd.DataFrame):
    data['year'] = data.index.year
    Benchmark = data.groupby('year')['d_ret'].sum()
    Strategy = data.groupby('year')['return_a'].sum()
    Diff = Strategy - Benchmark
    YR = pd.concat([Strategy, Benchmark, Diff], axis=1)
    YR.columns = ['Strategy', 'Benchmark', 'Diff']
    plt.bar(YR.index, YR['Strategy'], alpha=0.5, label='Strategy')
    plt.bar(YR.index, YR['Benchmark'], alpha=0.5, label='Benchmark')
    plt.legend()
    plt.show()
    return YR


'''畫出 Profit and Loss Curve'''
def PnL(data: pd.DataFrame):
    plt.plot(data['return_a'].cumsum(), label='Strategy')
    plt.plot(data['d_ret'].cumsum(), label='Benchmark')
    plt.ylabel('Cumulative Return (100%)')
    plt.legend()
    plt.show()
    plt.plot(data['return_a'].cumsum() - data['d_ret'].cumsum(), label='Difference of Strategy and Benchmark')
    plt.ylabel('Cumulative Return (100%)')
    plt.legend()
    plt.show()
    plt.plot(data['return_a'].cumsum(), label='Strategy')
    plt.plot(data['return_l'].cumsum(), label='PnL of Long')
    plt.plot(data['return_s'].cumsum(), label='PnL of Short')
    plt.plot(MDD_Series(data['return_a']), label='MDD')
    newhigh = pd.Series(np.where((data['return_a'].cumsum().cummax() - data['return_a'].cumsum() == 0), data['return_a'].cumsum(), np.nan), index=return_data.index)
    newhigh = newhigh.replace(0, np.nan).dropna()
    plt.scatter(newhigh.index, newhigh, color='lime', zorder=6, s=10)
    plt.ylabel('Cumulative Return (100%)')
    plt.legend()
    plt.show()


'''最大風險回撤函數'''
def MDD_Series(data: pd.DataFrame):
    return -(data.dropna().cumsum().cummax() - data.dropna().cumsum())


'''各種比率'''
class Ratio:
    def __init__(self, series: pd.DataFrame):
        self.data = series

    def Cummulative_Return(self):
        return self.data.dropna().cumsum()[-1]

    def Annual_Return(self):
        return ((1 + self.data.mean()) ** 252)-1

    def Annual_Volatility(self):
        return self.data.std() * (250 ** 0.5)

    def MDD_Return(self):
        MS = -MDD_Series(self.data)
        MDD = max(MS)
        return self.data.dropna().cumsum()[-1] / MDD

    def Sharpe(self):
        rf = (0.002358 + 1) ** (1/252) - 1
        return ((self.data.mean() - rf)/self.data.std()) * (252 ** 0.5)

    def Sortino(self):
        rf = (0.002358 + 1) ** (1/252) - 1
        return ((self.data.mean() - rf)/self.data[self.data < 0].std()) * (252 ** 0.5)


'''高原圖'''
class Optimization_Map:
    def __init__(self, mode):
        self.upwide = []
        self.dnwide = []
        self.z = []
        self.mode = str(mode)

    def CumRet(self, xfrom, xto, xby, yfrom, yto, yby):
        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_band = BBands(stock, self.mode, 20, i, j)
                temp_df = pd.concat([stock, temp_band], axis=1)
                temp_signal = Signal(temp_df)
                return_df = pd.concat([temp_df['Close'].pct_change(), temp_signal['all']], axis=1)
                return_df.columns = ['d_return', 'position_all']
                return_df['return_a'] = return_df['d_return'] * return_df['position_all']
                cumret = Ratio(return_df['return_a']).Cummulative_Return()
                self.upwide.append(i)
                self.dnwide.append(j)
                self.z.append(cumret)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.z, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('CumReturn')
        plt.show()

    def RonMDD(self, xfrom, xto, xby, yfrom, yto, yby):
        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_band = BBands(stock, self.mode, 20, i, j)
                temp_df = pd.concat([stock, temp_band], axis=1)
                temp_signal = Signal(temp_df)
                return_df = pd.concat([temp_df['Close'].pct_change(), temp_signal['all']], axis=1)
                return_df.columns = ['d_return', 'position_all']
                return_df['return_a'] = return_df['d_return'] * return_df['position_all']
                temp_MDD = Ratio(return_df['return_a']).MDD_Return()
                self.upwide.append(i)
                self.dnwide.append(j)
                self.z.append(temp_MDD)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.z, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('RonMMD')
        plt.show()

    def Sharpe_Ratio(self, xfrom, xto, xby, yfrom, yto, yby):
        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_band = BBands(stock, self.mode, 20, i, j)
                temp_df = pd.concat([stock, temp_band], axis=1)
                temp_signal = Signal(temp_df)
                return_df = pd.concat([temp_df['Close'].pct_change(), temp_signal['all']], axis=1)
                return_df.columns = ['d_return', 'position_all']
                return_df['return_a'] = return_df['d_return'] * return_df['position_all']
                temp_sharpe = Ratio(return_df['return_a']).Sharpe()
                self.upwide.append(i)
                self.dnwide.append(j)
                self.z.append(temp_sharpe)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.z, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Sharpe Ratio')
        plt.show()

    def Sortino_Ratio(self, xfrom, xto, xby, yfrom, yto, yby):
        for i in np.arange(xfrom, xto, xby):
            for j in np.arange(yfrom, yto, yby):
                temp_band = BBands(stock, self.mode, 20, i, j)
                temp_df = pd.concat([stock, temp_band], axis=1)
                temp_signal = Signal(temp_df)
                return_df = pd.concat([temp_df['Close'].pct_change(), temp_signal['all']], axis=1)
                return_df.columns = ['d_return', 'position_all']
                return_df['return_a'] = return_df['d_return'] * return_df['position_all']
                temp_sortino = Ratio(return_df['return_a']).Sortino()
                self.upwide.append(i)
                self.dnwide.append(j)
                self.z.append(temp_sortino)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self.upwide, self.dnwide, self.z, cmap=plt.cm.viridis, linewidth=0.2)
        ax.set_xlabel('UPwide')
        ax.set_ylabel('Dnwide')
        ax.set_zlabel('Sortino Ratio')
        plt.show()


def Monte_Carlo_Simulation(data: pd.DataFrame, n):
    for i in range(1, n, 1):
        random_position = pd.DataFrame(data['position_a'].sample(frac=1)).set_index(data.index)
        df = pd.concat([data['d_ret'], random_position], axis=1)
        df['new_ret'] = df['d_ret'] * df['position_a']
        plt.plot(df['new_ret'].cumsum())
    plt.plot(data['return_a'].cumsum(), color='black', linewidth=3.0)
    plt.show()



if __name__ == '__main__':
    stock = YahooData('2609.TW', '2010-01-01', '2021-09-01')
    mode = 'o'
    bands = BBands(stock, mode, 30, 2, 2)
    mix_data = pd.concat([stock, bands], axis=1)
    sig = Signal(mix_data)
    return_data = Returns(mix_data, sig)
    print(Yearly_return(return_data))
    PnL(return_data)
    print(return_data['return_a'].mean())
    print(return_data['return_a'].std())


    print('=========================Strategy Ratio==================================')
    print('累積日報酬率：', Ratio(return_data['return_a']).Cummulative_Return()*100, '%')
    print('年化報酬率：', Ratio(return_data['return_a']).Annual_Return() * 100, '%')
    print('年化波動率：', Ratio(return_data['return_a']).Annual_Volatility() * 100, '%')
    print('最大風險回撤：', max(-MDD_Series(return_data['return_a']))*100, '%')
    print('風報比：', Ratio(return_data['return_a']).MDD_Return())
    print('夏普率：', Ratio(return_data['return_a']).Sharpe())
    print('索丁諾比率：', Ratio(return_data['return_a']).Sortino())
    print('=========================Benchmark Ratio==================================')
    print('累積日報酬率：', Ratio(return_data['d_ret']).Cummulative_Return() * 100, '%')
    print('年化報酬率：', Ratio(return_data['d_ret']).Annual_Return() * 100, '%')
    print('年化波動率：', Ratio(return_data['d_ret']).Annual_Volatility() * 100, '%')
    print('最大風險回撤：', max(-MDD_Series(return_data['d_ret'])) * 100, '%')
    print('風報比：', Ratio(return_data['d_ret']).MDD_Return())
    print('夏普率：', Ratio(return_data['d_ret']).Sharpe())
    print('索丁諾比率：', Ratio(return_data['d_ret']).Sortino())


    Optimization_Map(mode).CumRet(0.25, 3.25, 0.25, 0.25, 3.25, 0.25)
    Optimization_Map(mode).RonMDD(0.25, 3.25, 0.25, 0.25, 3.25, 0.25)
    Optimization_Map(mode).Sharpe_Ratio(0.25, 3, 0.25, 0.25, 3.25, 0.25)
    Optimization_Map(mode).Sortino_Ratio(0.25, 3, 0.25, 0.25, 3.25, 0.25)


    Monte_Carlo_Simulation(return_data, 50)







