import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pyfolio as pf
import matplotlib.pyplot as plt
import streamlit as st

class Setup:
    def __init__(self, stocks, days):
        self.days = days
        self.stocks = stocks

    def data_pull(self):
        today = dt.datetime.today()
        start = today - dt.timedelta(self.days)
        tickers = yf.Tickers(self.stocks)
        stocks_data = tickers.history(start=start, end=today)
        self.data = stocks_data['Close']
        return self.data

    def calc_johansen(self):
        result = coint_johansen(self.data, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1
        crit_value = result.cvt[:, 1]
        return trace_stat, crit_value

    def calc_spread(self): #eliminate lookback bias here
        self.stock1_data = self.data.iloc[:, 0]
        self.stock2_data = self.data.iloc[:, 1]
        ln_stock_1 = np.log(self.stock1_data)
        ln_stock_2 = np.log(self.stock2_data)
        stock1_c = sm.add_constant(ln_stock_1)
        results = sm.OLS(ln_stock_2, stock1_c).fit()

        b = results.params[1]
        spread = ln_stock_2 - b * ln_stock_1
        r_mean = spread.rolling(window=30).mean()
        r_sd = spread.rolling(window=30).std()
        zscore = (spread - r_mean) / r_sd
        return zscore


class TradeBacktest:
    def __init__(self, data, zscore):
        self.port = pd.DataFrame(data, index=data.index)
        self.port['zscore'] = zscore
        self.stock1_data = data.iloc[:, 0]
        self.stock2_data = data.iloc[:, 1]

    def backtest(self):
        self.port['s1_return'] =self. stock1_data.pct_change()
        self.port['s2_return'] = self.stock2_data.pct_change()
        self.port['spread'] = self.stock1_data - self.stock2_data

        #Backtesting Framework#
        self.port['signal'] = 2
        self.port['signal'].iloc[0] = 0

        self.port.loc[self.port['zscore'] > 1.5, 'signal'] = -1
        self.port.loc[self.port['zscore'] < -1.5, 'signal'] = 1
        self.port.loc[(self.port['zscore'] >= -0.5) & (self.port['zscore'] <= 0.5), 'signal'] = 0
        
        self.port['position'] = self.port['signal']
        for i in range(len(self.port)):
            if self.port['position'].iloc[i] == 2:
                self.port.at[self.port.index[i], 'position'] = self.port['position'].iloc[i - 1]
            else:
                self.port.at[self.port.index[i], 'position'] = self.port['signal'].iloc[i]

        self.port['position'] = self.port['position'].shift(1)
        self.port['strategy_return'] = self.port['position'] * (self.port['s1_return'] - self.port['s2_return'])
        self.port['strategy_return'] = self.port['strategy_return'].fillna(0)
        self.port['cumulative_return'] = (self.port['strategy_return'] + 1).cumprod()
        return self.port['strategy_return']

    @staticmethod
    def risk_analysis(strategy_returns):
        pyfolio_stats = pf.timeseries.perf_stats(returns=strategy_returns)
        return pyfolio_stats

    @staticmethod
    def plot(strategy_returns):
        fig_names = ['cumulative_returns.png', 'rolling_sharpe.png', 'drawdown.png']

        #returns
        plt.figure(figsize=(14, 8))
        pf.plotting.plot_rolling_returns(strategy_returns)
        plt.savefig(fig_names[0])
        plt.close()
        
        #Rolling Sharpe
        plt.figure(figsize=(14, 8))
        pf.plotting.plot_rolling_sharpe(strategy_returns)
        plt.savefig(fig_names[1])
        plt.close()

        #Drawdown
        plt.figure(figsize=(14, 8))
        pf.plotting.plot_drawdown_periods(strategy_returns)
        plt.savefig(fig_names[2])
        plt.close()

        return fig_names


if __name__ == '__main__':
    #Streamlit#
    st.title("Pairs Trading Analysis")

    with st.sidebar:
        st.title('Inputs:')
        st.write("Input two stock tickers to test cointegration and backtest a basic trading strategy")
        stock_input1 = st.text_input("Ticker 1: ", value='HD')
        stock_input2 = st.text_input("Ticker 2: ", value='LOW')
        stocks = [stock_input1, stock_input2]
        data_days = st.number_input("Days for historical data", value=365)
        button = st.button('Run Analysis')

        st.markdown("#### Personal Information: ")
        linkedin_url = 'https://www.linkedin.com/in/michaelwojciechowski93'
        st.markdown(
            f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Michael Wojciechowski</a>',
            unsafe_allow_html=True)

    if button == True:
        setup = Setup(stocks=stocks, days=data_days)
        data = setup.data_pull()
        trace, crit = setup.calc_johansen()
        if any(trace > crit):
            zscore = setup.calc_spread()

            trade = TradeBacktest(zscore=zscore, data=data)
            strat_returns = trade.backtest()
            risk_stats = trade.risk_analysis(strat_returns)

            st.markdown("### Backtest Statistics: ")
            col1, col2, col3 = st.columns(3)
            col1.metric(label='Annual return', value=f"{round(risk_stats[0] * 100,2)}%")
            col2.metric(label='Cumulative returns', value=f"{round(risk_stats[1] * 100,2)}%")
            col3.metric(label='Annual volatility', value=f"{round(risk_stats[2] * 100,2)}%")
            col1.metric(label='Sharpe ratio', value=f"{round(risk_stats[3],2)}")
            col2.metric(label='Max drawdown', value = f"{round(risk_stats[6] * 100,2)}%")
            col3.metric(label= 'Daily value at risk', value=f"{round(risk_stats[12] * 100,2)}%")

            #Chart
            st.markdown("### Backtest Plots: ")
            fig_names = trade.plot(strat_returns)
            for fig_name in fig_names:
                st.image(fig_name, caption=fig_name)

        else:
            st.markdown("# Stocks Are Not Cointegrated! ")
            st.write("Please try two other pairs, or select xyz to find cointegrated pairs from a preselected universe of stocks")

        #Model Description
        st.markdown('#### About the Project')
        with st.expander("Click Here To Read More About The Project:"):
            st.write("xyz model description")
