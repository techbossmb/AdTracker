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
        if negative_size > positive_size:
            negative_df = negative_df.sample(positive_size)
        df = pd.concat([positive, negative_df])
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