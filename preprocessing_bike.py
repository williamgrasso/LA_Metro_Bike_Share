from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from datetime import date, datetime
import numpy as np


"""

"""
class Preprocess_bike:
   
    def __init__(self):
        pass
        
    
    def fit(self, X, y=None):
        return self
    
    #function to get season based on date
    def get_season(self, now):
        Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
        seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
                ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
                ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
                ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
                ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]
        now = now.replace(year=Y)
        return next(season for season, (start, end) in seasons
                    if start <= now <= end)
    
    #implement get if date is holiday, convert date format, merge holiday column, get season and encode all
    def transform(self, X, y=None):
        
        #import holiday calendar
        cal = calendar()
        holidays = cal.holidays(start='2016-01-01', end='2019-12-31')#.to_pydatetime()
        holidays = pd.DataFrame(holidays)
        holidays = holidays.rename(columns={0: 'holidays'})
        
        #convert date format
        X['DATE'] = pd.to_datetime(X['DATE'])
        X['Day_of_Week'] = X['DATE'].dt.dayofweek
        X['Day_of_Year'] = X['DATE'].dt.dayofyear
        X['Month'] = X['DATE'].dt.month 
        
        #convert trip ID string to int
        #X['Count of Trip Id'] = X['Count of Trip Id'].str.replace(',', '').astype(int)
        
        X = pd.merge(X, holidays, left_on='DATE', right_on ='holidays', how = 'left')
        #X = X.rename(columns={'Day_of_Date': 'Day_of_Date'})
        
        
        #season
        X['season'] = X.DATE.map(self.get_season)

        #Encode season    
        X.loc[X['season'] == 'winter', 'winter'] = 1
        X.loc[X['season'] == 'spring', 'spring'] = 1
        X.loc[X['season'] == 'summer', 'summer'] = 1
        X.loc[X['season'] == 'autumn', 'autumn'] = 1

        X['winter'] = X['winter'].fillna(0)
        X['spring'] = X['spring'].fillna(0)
        X['summer'] = X['summer'].fillna(0)
        X['autumn'] = X['autumn'].fillna(0)

        #Fill holiday NaT with 0
        X['holidays'] = X['holidays'].fillna(0)

        #Create column for 0 non-holiday and 1 holiday
        X.loc[X['holidays'] == 0, 'holiday'] = 0
        X['holiday'] = X['holiday'].fillna(1)


        X.drop(["holidays", "season"], axis = 1, inplace = True)
        
        #z_score
        count_mean = X['Count_of_Trip_Id'].mean()
        count_std = X['Count_of_Trip_Id'].std()
        X['count_z_score'] = (X['Count_of_Trip_Id'] - count_mean) / count_std
        
        
        
        #One Hot encode months and days of week
        X_one_hot = pd.get_dummies(X.Month, prefix='Month')
        X_one_hot_day_of_week = pd.get_dummies(X.Day_of_Week, prefix='day_of_week')
     
        #Rejoin one hot encoded months
        X =pd.concat([X, X_one_hot,X_one_hot_day_of_week], axis=1, sort=False)
        
        #Get month counts
#         Month_count = X.groupby('Month').sum()
#         Month_count = X['Count_of_Trip_Id']
#         Month_count = pd.DataFrame(Month_count)
#         Month_count = Month_count.rename(columns = {'Count_of_Trip_Id': 'Month_Count_of_Trip_Id'})
#         X = pd.merge(X, Month_count, left_on='Month', right_on ='Month', how = 'left')
       
        
        
        return X
    
    def fit_transform(self,X, y=None):
        self.fit(X,y)
        return self.transform(X, y) 

    
