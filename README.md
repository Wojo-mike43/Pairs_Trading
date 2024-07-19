# Pairs_Trading
Streamlit App Link:
https://pairstrading-48su4mxdvvjsmkt5w2chmc.streamlit.app

This project is a statistical-arbitrage pairs trading project that allows the user to select two stocks and the number of days to compare the stocks.

The script collects daily historical closing price data from Yahoo Finance and tests the cointegration between the two equities utilizing a Johansen test. A 95% critical value is used as the cut-off for a pair to be considered cointegrated.

If the stocks are found to be cointegrated, the spread between the two stocks is calculated. This involves using logarithmic transformation to ensure the data is log-normal, adding a constant to one of the stocks, and fitting a linear regression. The resulting beta coefficient is then used to calculate the spread, and the rolling mean and standard deviation of the spread are calculated. Rolling statistics are used to eliminate lookback bias. Z-scores for the spread are then returned.

Z-scored are utilized to generate trading signals, shorting the spread when the z-score is 1.5 or higher, and going long the spread when its z-score is -1.5 or lower. Positions are exited if they enter the z-score range of -0.5 to 0.5. 

The strategy is back-tested using a Pandas DataFrame by assigning trading signal values to a data frame column named “signals,” and then iterating over the column to create position values. Position values are shifted to account for trades taking place the next trading day. Strategy and cumulative returns are calculated to analyze strategy.

Pyfolio is utilized to produce strategy statistics such as Annual Return, Sharpe Ratio, and Max drawdown. Pyfolio also produces plot figures for returns and rolling Sharpe Ratio, which is presented on the Streamlit GUI page. 

