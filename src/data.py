import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def get_data_dropNaN(df):
    ''' takes df of imported data
    Cleans and drops all columns with Nan values
    returns X: df of predictor variable
            y: df of response variables'''

    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])

    cols_to_keep = ['avg_dist', 'avg_surge','last_trip_date', 'signup_date', 'surge_pct',
       'trips_in_first_30_days', 'luxury_car_user', 'weekday_pct']
    df = df.filter(cols_to_keep)
    df['luxury_car_user'] = df['luxury_car_user'].astype(int)


    churn_date = datetime.date(2014,6,1)

    y = df.last_trip_date < churn_date
    y = y.astype(int)

    df.drop(['last_trip_date','signup_date'],axis = 1,inplace=True)

    return df,y



def get_data(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['luxury_car_user'] = df['luxury_car_user'].astype(int)
    df = pd.get_dummies(df)
    rate_driver = ~df.avg_rating_of_driver.isnull()
    df['rate_driver'] = rate_driver.astype(int)
    df.drop('avg_rating_of_driver',axis = 1, inplace= True)
    gets_rated = ~df.avg_rating_by_driver.isnull()
    df['gets_rated'] = gets_rated.astype(int)
    df.drop('avg_rating_by_driver',axis = 1, inplace= True)

    churn_date = datetime.date(2014,6,1)

    y = df.last_trip_date < churn_date
    y = y.astype(int)

    df.drop(['last_trip_date','signup_date'],axis = 1,inplace=True)

    return df,y


if __name__ == '__main__':
    df = pd.read_csv('../data/churn_train.csv')
    X,y = get_data_dropNaN(df)
