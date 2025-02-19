import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

d=yaml.safe_load(open('params.yaml','r'))['data_ingestion']

def load_data(url):
    df=pd.read_csv(url)
    return df

def train_test_data(df):
    train_data,test_data=train_test_split(df,test_size=d['test_size'],random_state=d['random_state'])
    return train_data,test_data

def save_data(train_data,test_data,data_path):
    os.makedirs(data_path)
    train_data.to_csv(os.path.join(data_path,'train.csv'))
    test_data.to_csv(os.path.join(data_path,'test.csv'))

def main():
    df=load_data('C:/Users/ayush/OneDrive/Desktop/tipsdataset.csv')
    train_data,test_data=train_test_data(df)
    data_path='data/raw'
    save_data(train_data,test_data,data_path)

if __name__=='__main__':
    main()

