import pandas as pd
import numpy as np

def yield_dataset(file, batchsize):
    ''' 
    Dataset is very large can't be held in memory at once has to be loaded in bits
    train dataset is ~7.4GB
    Dataset also has huge class imbalance, so we rebalance the class ratio
    '''
    imbalance_dfs = pd.read_csv(file, iterator=True, chunksize=batchsize)
    
    for imbalance_df in imbalance_dfs:
        positive_df = imbalance_df[imbalance_df.is_attributed==1]
        positive_size = positive_df.shape[0]
        
        # if no positive label in current batch, disregard the whole batch data
        if positive_size == 0:
            continue
        
        negative_df = imbalance_df[imbalance_df.is_attributed==0]
        negative_size = negative_df.shape[0]
        
        # generate equal negative and positive samples by downsampling the negative class
        if negative_size > positive_size:
            negative_df = negative_df.sample(positive_size)

        df = pd.concat([positive_df, negative_df])
        df = df.drop(['click_time', 'attributed_time'], axis=1)
        features = np.asarray(df.drop(['is_attributed'], axis=1))
        features = features.reshape(features.shape[0], features.shape[1],1)
        labels = np.asarray(df['is_attributed'])
        yield features, labels

def datagenerator(file, batchsize):
    generator = yield_dataset(file, batchsize)
    while True:
        try:
            current_data = next(generator)
            yield current_data
        except StopIteration:
            generator = yield_dataset(file, batchsize)

def load_dataset(file):
    df = pd.read_csv(file)
    df = feature_transformation(df)
    feature_df = df.drop(['attributed_time', 'is_attributed'], axis=1)
    feature_df = feature_df.drop(['click_time'], axis=1)
    label_df = df['is_attributed']
    return feature_df, label_df

def feature_transformation(dataframe):
    '''
    perform feature extraction and create more useful features
    '''
    dataframe['hour'] = pd.to_datetime(dataframe.click_time).dt.hour
    dataframe['day'] = pd.to_datetime(dataframe.click_time).dt.day
    # bin hours
    start_hour_bins = [0, 6, 12, 18]
    end_hour_bins = [6, 12, 18, 24]
    hour_bins = [0,1,2,3]
    for i, (start_hour, end_hour) in enumerate(zip(start_hour_bins, end_hour_bins)):
        dataframe.loc[(dataframe.hour >= start_hour) & (dataframe.hour < end_hour), 'hour_bin'] = hour_bins[i]
    
    # ip_click freq - num of click associated with each unique ip
    clicks_per_ip = dataframe[['ip','channel']]\
        .groupby(by=['ip'])[['channel']].count().reset_index().rename(columns={'channel': 'clicks_per_ip'})
    dataframe = dataframe.merge(clicks_per_ip, on=['ip'], how='left')

    # ip_channel freq per hour - num of channels associated with each unique ip per hour
    channels_per_ip_hour = dataframe[['ip','day','hour','channel']]\
        .groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(columns={'channel': 'channels_per_ip_hour'})
    dataframe = dataframe.merge(channels_per_ip_hour, on=['ip','day','hour'], how='left')

    # ip_app_channel freq - number of channels associated with each unique ip and app
    channels_per_ip_app = dataframe[['ip','app', 'channel']]\
        .groupby(by=['ip','app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
    dataframe = dataframe.merge(channels_per_ip_app, on=['ip','app'], how='left')
    
    return dataframe