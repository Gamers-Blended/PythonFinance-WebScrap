# Alt + 3 to block comment
# Alt + 4 to block uncomment

# pricing data to % change -> normalised
# parameters for machine learning:
# feature = % change
# labels = buy, sell, hold

## features defines something
## label is the target

# Qn: Did price based on features within next 7 trading days change by > 2%?
# cases:
# rise > 2% -> buy
# fall > 2% -> sell
# neither -> hold

# this file is to process data for a specific ticker

########################################################################################################################################################################

from collections import Counter
import numpy as np
import pandas as pd
import pickle # to serialise any python object, to save any object as a variable

#from sklearn.model_selection import cross_validate # (line no longer works)

from sklearn import svm, model_selection as cross_validation, neighbors
# cross_validation: creates training and testing samples (don't want to test against the same sample you trade against, testing should be blind with shuffle data)
# neighbors: to do K nearest neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
# VotingClassifer: since using many classifers and letting them vote on what they think is best

########################################################################################################################################################################

# each model generated is a per company basis, each company takes into account pricing data of all other companies in S&P500
def process_data_for_labels(ticker):
    hm_days = 7 # how many days in future to make/loss X%?
    df = pd.read_csv('sp500_joined_closes.csv', index_col = 0) # reads only from this specific csv file
    tickers = df.columns.values.tolist() 
    df.fillna(0, inplace = True)

    for i in range(1, hm_days+1): # need to start from 1 till end+1 since range starts from 0 by default
        #print(i) # to check
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        # custom data frame column name
        # eg: ExxonMobil on day 2: EOM_2d
        # shift: up/down -> this case shift negatively (up) by i numbers to get future data
        # value = (future price - old price) / old price

    df.fillna(0, inplace = True)

    return tickers, df
    
#process_data_for_labels('XOM')

# function to map to pandas dataframe, and then to a new column
def buy_sell_hold(*args): # args lets you pass any number of parameters
    cols = [c for c in args] # make a new column for target: buy, sell, hold
    # with mapping to pandas, function goes row by row in data frame, can pass nothing or columns as parameters
    requirement = 0.025 # if stock price change by 2.5% in 7 days
    for col in cols: # pass each column (tomorrow's price, the following day's price, so on for the whole week of future % changes)
        if col > requirement: # at any point in any of these columns, if value > requirement
            return 1 # 1 = buy
        if col < -requirement:
            return -1 # 1 = sell
    return 0 # 0 = hold
    
# idea:
# if this can possibly rise > 2.5% within the next 7 days -> buy
# then sell at +1.5% of price paid
# stop loss at -2%/-1.5%
# if price going to fall > 2.5% within the next 7 days -> don't invest right now

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker) # tickers are the prices
    # define new column
    # value for this column will be the mapped answers of buy_sell_hold()
    # args* = 7 days % change in price

    # creates a column and generates buy/sell/hold
    df['{}_target'.format(ticker)] = list(map( buy_sell_hold, # eg: XOM_target
                                   df['{}_1d'.format(ticker)], # map(<FUNCTION>, <PARAMETERS TO PASS TO THIS FUNCTION>)  
                                   df['{}_2d'.format(ticker)],
                                   df['{}_3d'.format(ticker)],
                                   df['{}_4d'.format(ticker)],
                                   df['{}_5d'.format(ticker)],
                                   df['{}_6d'.format(ticker)],
                                   df['{}_7d'.format(ticker)]
                                               ))
##    hm_days = 7
##    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *   [df['{}_{}d'.format(ticker,i)] for i in range(1,hm_days+1)]))
    
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:' , Counter(str_vals)) # Counter() require string, displays number of tickers in index to buy/sell/hold, more balanced the numbers -> more accurate

    df.fillna(0, inplace = True) # if have any prior na, call them 0

    df = df.replace([np.inf, -np.inf], np.nan) # replaces np.inf (any infinite changes when price goes from $0 to something) with np.nan
    df.dropna(inplace = True)

    # be explict to which columns get to be the feature sets
    df_vals = df[[ticker for ticker in tickers]].pct_change() # ticker prices normalised: today's value as opposed to yesterday (% change from yesterday)    
    df_vals = df_vals.replace([np.inf, -np.inf], 0) # fills infinites with 0
    df_vals.fillna(0, inplace = True) # fills na with 0

    # feature sets (things that describle labels) (daily price (%) changes) = X
    # labels (target, class) = y
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df, #hm_days

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker) # feature sets and labels

    # training and testing
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                         y,
                                                                         test_size = 0.25) # 25% of sample data will be tested against
    # defines classifier
    #clf = neighbors.KNeighborsClassifier()

    # takes 3 classifers and let them vote
    # takes a list of tuples of 3 classifiers by name in actual classifier
    clf = VotingClassifier([('lsvc', svm.LinearSVC()), # linear support vector classifier, referencing support vector machine for linear support vector classification, default parameters
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    # use a classifier that will fit input data to target
    clf.fit(X_train, y_train)
    # X_train = % change data for all companies including company in question
    # y = target: 0,1,-1
    confidence = clf.score(X_test, y_test)
    print('Accurary', confidence) # prints accuracy

    # if you train and happy with the confidence, to further predict, do clf.predict(X_test)
    # if not want to retrain, pickle out classifier
    # to use classifier again, load in classifier (call clf, clf.predict(X_test))
    
    predictions = clf.predict(X_test) # outputs values
    print('Predicted spread:', Counter(predictions)) # displays number of votes to buy/sell/hold    
    return confidence

    # issues with classifiers: overfitting, everything becomes 1 class
    # happens often within balanced data

do_ml('BAC') # insert ticker
    



