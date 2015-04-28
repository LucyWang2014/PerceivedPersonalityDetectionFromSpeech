# -*- coding: utf-8 -*-
"""
Lucy Wang
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

os.chdir("/Users/Lucy/Google Drive/DSGA1003 - project")


#split dataset into X and Y
def splitXY(filepath):
    Target_Col = ['Aggressive','Attractive','Confident','Intelligent','Masculine','Quality','Trust','Win']    
    
    DF = pd.read_csv(filepath,index_col=0)    
    
    DF_y = DF[Target_Col]
    DF_y = pd.concat([DF.ix[:,['WorkerId']],DF_y],axis=1,join_axes=[DF.index])
    DF_x = DF.drop(Target_Col, axis=1)
    
    return DF_x,DF_y
    
#transform target data

def TransformTarget(DF_y):    
    DF_y = DF_y.fillna(DF_y.mean())
    
    cols = list(DF_y.columns)
    cols.remove('WorkerId')
    
    for col in cols:
        col_zscore = col + '_zscore'
        DF_y[col_zscore] = DF_y[['WorkerId',col]].groupby('WorkerId').transform(lambda x: (x - x.mean()) / x.std(ddof=1))
    
    DF_y = DF_y.fillna(0)
    
    for col in cols:
        col_label = col + '_label'
        col_zscore = col + '_zscore'
        DF_y[col_label] = 0
        DF_y.ix[DF_y[col_zscore] > 0,col_label] = 1  
        
    Target_Col = ['WorkerId','Aggressive_label','Attractive_label','Confident_label','Intelligent_label','Masculine_label','Trust_label']    
    
    DF_y = DF_y[Target_Col]
    
    return DF_y    

def SplitTrainTest(DF_x,DF_y,train_percent = 0.75):
    DF_Index = pd.read_csv("Data/Intermediary Datasets/WorkerId.csv")
    Num_Rows = np.shape(DF_Index)[0] 
    train_index = int(Num_Rows * train_percent)
    
    index = np.arange(0,Num_Rows)
    np.random.shuffle(index)
    
    DF_Index["Index"] = index
    DF_Index = DF_Index.sort(["Index"])
    
    DF_x = pd.merge(DF_x, DF_Index, on='WorkerId')
    DF_y = pd.merge(DF_y, DF_Index, on='WorkerId')
    
    drop_col = ['WorkerId','Index']
    
    DF_train_x = DF_x.loc[DF_x.Index <= train_index]
    DF_test_x = DF_x.loc[DF_x.Index > train_index].drop(drop_col,axis=1)
    DF_train_y = DF_y.loc[DF_y.Index <= train_index]
    DF_test_y = DF_y.loc[DF_y.Index > train_index].drop(drop_col,axis=1)
    
    return DF_train_x,DF_test_x,DF_train_y,DF_test_y
    
def wrapper(func,*args, **kwargs):
    return func(*args,**kwargs)

def CrossVal(DF_train_x,DF_train_y,func,k=10):
    DF_WorkerId=pd.Series(DF_train_x.WorkerId.values.ravel()).unique()     
    Num_Rows = DF_WorkerId.shape[0] 
    Num_Cols = DF_train_y.shape[1]-2
    
    kf = cross_validation.KFold(Num_Rows, n_folds=k,shuffle=True)
    
    score = np.zeros((Num_Cols,k,len(func)))
    
    x_cols = list(DF_train_x.columns)
    y_cols = list(DF_train_y.columns)  
    
    target_cols = y_cols[1:-1]
    
    k_fold=0
    for train_index, test_index in kf:
        DF_WorkerId_train = DF_WorkerId[train_index]
        DF_WorkerId_test = DF_WorkerId[test_index]       
        
        DF_xtrain_x = DF_train_x[DF_train_x.WorkerId.isin(DF_WorkerId_train)][x_cols[3:-1]]
        DF_xtrain_y = DF_train_y[DF_train_y.WorkerId.isin(DF_WorkerId_train)][y_cols[1:-1]]     
        DF_xtest_x = DF_train_x[DF_train_x.WorkerId.isin(DF_WorkerId_test)][x_cols[3:-1]]    
        DF_xtest_y = DF_train_y[DF_train_y.WorkerId.isin(DF_WorkerId_test)][y_cols[1:-1]]
        
        for i in range(len(target_cols)):
            for f in range(len(func)):
                func[f].fit(DF_xtrain_x,DF_xtrain_y[[i]].squeeze())
                score[i][k_fold][f] = func[f].score(DF_xtest_x,DF_xtest_y[[i]].squeeze())
                print "c = %s and target variable " % f + target_cols[i] + ": %s" % score[i][k_fold][f]
        k_fold += 1
    
    return score
    
def testScore(clf,DF_train_x,DF_train_y,DF_test_x,DF_test_y):
    Num_Cols = DF_test_y.shape[1]    
    score = np.zeros(Num_Cols)
    
    x_cols = list(DF_train_x.columns)
    y_cols = list(DF_train_y.columns)  
    
    DF_train_x = DF_train_x[x_cols[3:-1]]
    DF_train_y = DF_train_y[y_cols[1:-1]]     
    DF_test_x = DF_train_x[x_cols[3:-1]]    
    DF_test_y = DF_train_y[y_cols[1:-1]]    
    
    for i in range(Num_Cols):
        clf.fit(DF_train_x,DF_train_y[[i]].squeeze())
        score[i]=clf.score(DF_test_x,DF_test_y[[i]].squeeze())
        
    return score

#svm


def main():
    filepath = "Data/Final_Data/Master_4.26.15.csv"
    # there are 835 assessors, each evaluated approximately 66 recordings
    
    DF_x,DF_y = splitXY(filepath)
    
    DF_x = DF_x.fillna(-1)    
    
    DF_y.isnull().sum() #only 5 missing values
    
    DF_y = TransformTarget(DF_y)  
    
    
    DF_train_x,DF_test_x,DF_train_y,DF_test_y = SplitTrainTest(DF_x,DF_y,train_percent = 0.75)
    
    RF_func = []    
    nEstimators = [10,100,200,500,1000]
    for i in nEstimators:
        RF_func.append(wrapper(RandomForestClassifier,n_estimators=i))
    
    RF_Score = CrossVal(DF_train_x,DF_train_y,RF_func,k=2)
    
    Logit_func = []
    C = [10**x for x in range(-5,5)]
    for i in C:
        Logit_func.append(wrapper(LogisticRegression,C=i))
    
    Logit_Score = CrossVal(DF_train_x,DF_train_y,Logit_func,k=2)
    
    Logit_score_avg = np.mean(Logit_Score, axis=1)
    
    target_cols = list(DF_test_y.columns)
    position = 231    
    fig = plt.figure()
    for i in range(len(target_cols)):
        ax = fig.add_subplot(position)
        ax.plot(np.log(C),Logit_score_avg[i])
        ax.set_title(target_cols[i])
        ax.set_autoscaley_on(True)
        position += 1
    plt.tight_layout()
    fig.savefig("logit_plots.png")
    
    RF_score_avg = np.mean(RF_Score, axis=1)
    position = 231    
    fig = plt.figure()
    for i in range(len(target_cols)):
        ax = fig.add_subplot(position)
        ax.plot(np.log(nEstimators),RF_score_avg[i])
        ax.set_title(target_cols[i])
        ax.set_autoscaley_on(True)
        position += 1
    plt.tight_layout()
    fig.savefig("rf_plots.png")
        
    
    
    

