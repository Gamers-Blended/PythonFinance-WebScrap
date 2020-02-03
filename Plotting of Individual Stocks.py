# pandas
# pandas-datareader
# beautifulsoup4 - web scrapping library
# scikit-learn

import datetime as dt #sets starting and end dates for data to pull
import matplotlib.pyplot as plt # plots
from matplotlib import style #graphs look good
from mpl_finance import candlestick_ohlc #wants dates open high low close
import matplotlib.dates as mdates #matplotlib doesn't use daytime dates
import pandas as pd #data analysis library
import pandas_datareader.data as web #grabs from Yahoo Finance API, rtn pandas dataframe

#to remove warnings from conversion of time
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

style.use('ggplot')

# comment block: highlight -> Alt+3

##start = dt.datetime(2000,1,1)
##end = dt.datetime(2016,12,31)
##
###df = dataframe
###ticker, where to get from, what time
##df = web.DataReader('TSLA', 'yahoo', start, end)
### print(df.head()) #prints first 5 rows of df by default, add num inside bracket
###df.tail() last rows
##
##df.to_csv('tsla.csv')

#df = pd.read_csv('tsla.csv')
# print(df.head()) #shows the index numbers 0,1,2,... on the first column

####to get date-time index
df = pd.read_csv('tsla.csv', parse_dates = True, index_col = 0)
#print(df.head())

####dataframe objects have plot attributes
##df.plot() #shows all columns
##plt.show()

####to plot specific columns, use index
#df['Adj Close'].plot() #or print(df['Adj Close'])
#plt.show()

####possible to print multiple columns
#print(df[['Open', 'High']].head())

#100-day moving average
#df['100ma'] = df['Adj Close'].rolling(window=100).mean()

#inplace -> mod that dataframe as inplace as opposed to saying df = df.dropna
#don't have to redefine dataframe, can do it inplace
#entries with not a number are removed
#if not dropped, data starts 100 days earlier
df.dropna(inplace=True)
#df.tail will retrive last few data
#print(df.tail())

#to not use any number of period, don't need dropna
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods = 0).mean()
#print(df.head())

#each matplotlib has a figure
#figure contains all subplots
#1 subplot, 1 graph
#subplot = axes

#to generate subplots
#first parameter: grid size (number of rows, number of columns)
#second parameter: starting point
#third parameter: how many rows is this going to span?
#fourth parameter: how many columns is this going to span?
ax1 = plt.subplot2grid((6,1), (0,0) , rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0) , rowspan = 1, colspan = 1, sharex=ax1) #both graphs will have their own x-axes, same axe when zooming in

#to graph subplots
#x is the date,can reference to that index with df.index
##ax1.plot(df.index, df['Adj Close'])
##ax1.plot(df.index, df['100ma'])
##ax2.bar(df.index, df['Volume'])

##resampling
#eg: find number of people walking through door
#data is in per 3 minutes
#need to resample to 1 hour of data
#stock daily data -> 10 day data

#make new dataframe
#.mean() -> gets mean/average value over 10 days
#.sum() -> gets sum over 10 days
#.ohlc() -> creates data that are open high/low close
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum() #true volume, not average volume

#gets all contents of a dataframe, doesn't give column headers or index
#df.values

#reset index for df ohlc
df_ohlc.reset_index(inplace=True)


#have date, need to convert to end dates
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) #converts date/time object to mdates number 

#print(df_ohlc.head())

ax1.xaxis_date() #takes mdates and display them as beautiful dates

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
#parameters: (axes, data, thickness of candlesticks, colour)
#default colour: black up, red down
#candlestick graphs condense direction, open, high, low, close data in 1 line

#volume
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
# parameters: (x, y)

plt.show()

#grabbing S&P 500 data
#wikipedia page: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
#Ctrl+u to view source
#look for specific company
