import pandas as pd
import numpy as np
from ..signals import ContinuousSignal,CategoricalSignal
import random

def load_npz_data(dataset,fp='../datasets/npz/'):
    """
    load npz dataset
    """
    data = np.load(fp+dataset+'.npz', allow_pickle=True) 
    
    X = data['X']
    y = data['y']
    
    cols = []
    for i in range(X.shape[1]):
        cols.append('col'+str(i))
    df = pd.DataFrame(data=X,columns=cols)
    
    df['class'] = y
    df = df.sample(frac=1,random_state=1).reset_index(drop=True)
    
    pos = len(df)*4//5
    train_df = df.loc[:pos,:]
    train_df = train_df.loc[train_df['class']==0,:].reset_index(drop=True)
    test_df = df.loc[pos:,:].reset_index(drop=True)
    
    signals = []
    for name in train_df:
        if name!='class' and len(train_df[name].unique())>5:
            signals.append( ContinuousSignal(name, min_value=train_df[name].min(), max_value=train_df[name].max(),
                                            mean_value=train_df[name].mean(), std_value=train_df[name].std() ) )
        elif name!='class' and len(train_df[name].unique())<=5:
            signals.append( CategoricalSignal(name, values=train_df[name].unique().tolist()) )
    return train_df,test_df,test_df['class'].values,signals


def load_perturb_npz_data(dataset,fp='../datasets/npz/'):
    """
    load npz dataset
    """
    data = np.load(fp+dataset+'.npz', allow_pickle=True) 
    
    X = data['X']
    y = data['y']
    
    cols = []
    for i in range(X.shape[1]):
        cols.append('col'+str(i))
    df = pd.DataFrame(data=X,columns=cols)
    
    df['class'] = y

    df = df.sample(frac=1,random_state=1).reset_index(drop=True)
    df = df.loc[df['class']==0,:].reset_index(drop=True)
    
    pos = len(df)*4//5
    train_df = df.loc[:pos,:]
    test_df = df.loc[pos:,:].reset_index(drop=True)
    
    pdims = []
    for i in range(len(test_df)):
        pcount = random.randint(1, 3)
        dims = random.sample(range(0, len(cols)), pcount)
        pdims.append(dims)
        for d in dims:
            colname = 'col'+str(d)
            scale = train_df[colname].max()-train_df[colname].min()
            test_df.loc[i,colname] = train_df[colname].min() + random.random()* scale * 1.5

    signals = []
    for name in train_df:
        if name!='class' and len(train_df[name].unique())>5:
            signals.append( ContinuousSignal(name, min_value=train_df[name].min(), max_value=train_df[name].max(),
                                            mean_value=train_df[name].mean(), std_value=train_df[name].std() ) )
        elif name!='class' and len(train_df[name].unique())<=5:
            signals.append( CategoricalSignal(name, values=train_df[name].unique().tolist()) )   
    return train_df,test_df,pdims,signals