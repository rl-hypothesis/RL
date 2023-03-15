import pandas as pd
from numpy import ndarray

import config

def missing_genres(x, genres):
    if '(' in str(x.genres_first):
        g = str(x.genres).split(',')
        genr = ''
        for i in g:
            if i in genres:
                genr = genr+'|'+i
        return genr[1:]
    else:
        return str(x.genres_first)

def run(x):
    if x <= 1:
        return 'Short'
    elif x <= 3.5:
        return 'Long'
    else:
        return 'Very Long'

def name_groups(df):
    if (isinstance(df,int)) or (isinstance(df,float)) or (isinstance(df,str)):
        return str(df)
    
    #df = df.drop(columns=['cust_id','article_id','rating','purchase', 'transaction_date'])
    df = df.drop(columns=['article_id','rating','cust_id'])#.reset_index()
    #columns = list(df.index.names)

    columns = [col for col in df.columns if len(df[col].unique())==1]
    columns.sort()

    return ['_'.join(i) for i in df.reset_index()[columns].drop_duplicates().values][0]

def add_index(df):
    df['cust_index'] = df['cust_id'].astype('category').cat.codes
    df['cust_index'] = df['cust_index']+1

    df['article_index'] = df['article_id'].astype('category').cat.codes
    df['article_index'] = df['article_index']+1

    #cust_id2index = df[['cust_id','cust_index']].set_index('cust_id').T.to_dict('list')
    #cust_index2id = df[['cust_id','cust_index']].set_index('cust_index').T.to_dict('list')

    #article_id2index = df[['article_id','article_index']].set_index('article_id').T.to_dict('list')
    #article_index2id = df[['article_id','article_index']].set_index('article_index').T.to_dict('list')

    return df, {}, {}#cust_index2id, article_index2id
