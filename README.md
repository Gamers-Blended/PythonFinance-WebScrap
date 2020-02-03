# PythonFinance-WebScrap
Python for Finance

#1 Plotting of Individual Stocks
This file retrieves stock prices of a specific ticker (to be chosen by the user) from Yahoo Finance API, then stores them in a newly created CSV file.
It then plots a candlestick graph of the adjusted closing price of the selected stock.
TSLA is the default ticker.
A different stock can be selected by changing the ticker name in the web.DataReader() function.

#2 Web Scrap & Visualisation
A program used to scrap data from tables in websites and convert them into csv file.
It then visualises the data as a heatmap.
By default, data is scrapped from the S&P500 Wikipedia page.

#3 Preprocessing Data for Machine Learning
Uses the csv file generated from Web Scrap & Visualisation.py and votes buy/sell/hold decision through 3 classifiers:
svm.LinearSVC()
neighbors.KNeighborsClassifier()
RandomForestClassifier()
1 -> buy
-1 -> sell
0 -> hold

BAC is the default ticker.
