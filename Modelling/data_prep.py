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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

#set working directory
os.chdir("/Users/Lucy/MSDS/2015Spring/DSGA1003_Machine_Learning/project/codes")


#split dataset into the feature and target dataframes
def splitXY(filepath):
    '''
    splitXY splits the master dataframe into a feature dataframe and a label dataframe. Each dataframe will 
    have a workerID
    
    Args:
    filepath: the path to master dataset
    
    Returns:
    DF_x: dataframe containing all feature columns
    DF_y: dataframe containing all target columns
    '''
    
    Target_Col = ['Aggressive','Attractive','Confident','Intelligent','Masculine','Quality','Trust','Win']    
    
    DF = pd.read_csv(filepath,index_col=0)    
    
    DF_y = DF[Target_Col]
    DF_y = pd.concat([DF.ix[:,['WorkerId']],DF_y],axis=1,join_axes=[DF.index])
    DF_x = DF.drop(Target_Col, axis=1)
    
    return DF_x,DF_y
    
#transform target data

def TransformTarget(DF_y): 
    '''
    Transforms the target variables into binary labels.
    
    Every variable is z-score'd by WorkerID. This means the average of each quality for each Worker
    is rescaled to 0. Missing raw data are filled with mean of entire category. Null's from
    normalization are filled with 0.
    
    y > 0 -> 1
    y <= 0 -> 0
        
    Args:
    DF_y: dataframe of raw target variables
    
    Returns:
    DF_y: transformed target variables with binary value
    '''
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
    '''
    Splits the X and Y dataframes into training and test sets. The split is done by random sampling
    of WorkerId. 
    
    Args:
    DF_x: feature dataframe
    DF_y: transformed target variable dataframe
    train_percent: the amount of data to split for training
    
    Returns:
    DF_train_x:  feature dataframe for training 
    DF_test_x: feature dataframe for testing
    DF_train_y: target dataframe for training
    DF_test_y: target dataframe for testing
    '''
    DF_Index = pd.read_csv("Data/WorkerId.csv")
    Num_Rows = np.shape(DF_Index)[0] 
    train_index = int(Num_Rows * train_percent)
    
    index = np.arange(0,Num_Rows)
    np.random.shuffle(index)
    
    DF_Index["Index"] = index
    DF_Index = DF_Index.sort(["Index"])
    
    DF_x = pd.merge(DF_x, DF_Index, on='WorkerId')
    DF_y = pd.merge(DF_y, DF_Index, on='WorkerId')
    
    DF_train_x = DF_x.loc[DF_x.Index <= train_index]
    DF_test_x = DF_x.loc[DF_x.Index > train_index]
    DF_train_y = DF_y.loc[DF_y.Index <= train_index]
    DF_test_y = DF_y.loc[DF_y.Index > train_index]
    
    return DF_train_x,DF_test_x,DF_train_y,DF_test_y
    
def wrapper(func,*args, **kwargs):
    '''
    wrapper function for running models
    
    Args:
    func: the classification algorithm
    *args: args associated with the function
    **kwargs: keyword args associated with the function
    
    Returns:
    the classifier with specified parameters
    '''
    return func(*args,**kwargs)

def CrossVal(DF_train_x,DF_train_y,func,k=10):
    '''
    Parameter tuning with cross validation
    
    Args:
    DF_train_x: feature dataframe for training
    DF_train_y: transformed target dataframe for testing
    func: a list of classifier instances
    k: number of folds
    
    Returns:
    score: 3 dimensional list of scores for every target variable, every hyperparameter, and every fold
    '''
    DF_WorkerId=pd.Series(DF_train_x.WorkerId.values.ravel()).unique()     
    Num_Rows = DF_WorkerId.shape[0] 
    Num_Cols = DF_train_y.shape[1]-2
    
    kf = cross_validation.KFold(Num_Rows, n_folds=k,shuffle=True)
    
    score = np.zeros((Num_Cols,k,len(func)))
    
    x_cols = DF_train_x.columns.values.tolist()
    y_cols = DF_train_y.columns.values.tolist() 
    
    feature_cols = x_cols[3:-1]
    target_cols = y_cols[1:-1]
    
    k_fold=0
    for train_index, test_index in kf:
        DF_WorkerId_train = DF_WorkerId[train_index]
        DF_WorkerId_test = DF_WorkerId[test_index]       
        
        DF_xtrain_x = DF_train_x[DF_train_x.WorkerId.isin(DF_WorkerId_train)][feature_cols]
        DF_xtrain_y = DF_train_y[DF_train_y.WorkerId.isin(DF_WorkerId_train)][target_cols]     
        DF_xtest_x = DF_train_x[DF_train_x.WorkerId.isin(DF_WorkerId_test)][feature_cols]    
        DF_xtest_y = DF_train_y[DF_train_y.WorkerId.isin(DF_WorkerId_test)][target_cols]
        
        for i in range(len(target_cols)):
            for f in range(len(func)):
                func[f].fit(DF_xtrain_x,DF_xtrain_y[[i]].squeeze())
                score[i][k_fold][f] = func[f].score(DF_xtest_x,DF_xtest_y[[i]].squeeze())
                print "c = %s and target variable " % f + target_cols[i] + ": %s" % score[i][k_fold][f]
        k_fold += 1
    
    return score
    
def TrainTestClean(DF_train_x,DF_train_y,DF_test_x,DF_test_y):
    '''
    Take out all columns that are not part of the model, including index and identifier columns
    
    Args:
    DF_train_x: feature dataframe for training
    DF_train_y: target dataframe for training
    DF_test_x: feature dataframe for testing
    DF_test_y: taret dataframe for testing
    
    Returns:
    DF_train_x: training feature dataframe ready to put in model
    DF_train_y: training target dataframe ready to put in model
    DF_test_x: test feature dataframe ready to put in model
    DF_test_y: test target dataframe ready to put in model
    '''
    
    x_cols = DF_train_x.columns.values.tolist()
    y_cols = DF_train_y.columns.values.tolist()
    
    feature_cols = x_cols[3:-1]
    target_cols = y_cols[1:-1]
    
    DF_train_x = DF_train_x[feature_cols]
    DF_train_y = DF_train_y[target_cols]     
    DF_test_x = DF_test_x[feature_cols]    
    DF_test_y = DF_test_y[target_cols]  

    return DF_train_x, DF_train_y,DF_test_x,DF_test_y
    
def testScore(Models,DF_test_x,DF_test_y):
    '''
    Calculates the test score for each target variable using a list of models
    
    Args:
    Models: a list of fitted classifiers to test, same length as the number of target variables
    DF_test_x: test feature dataframe
    DF_test_y: test target dataframe
    
    Returns:
    score: an numpy array of the scores for each target variable
    '''
    Num_Cols = DF_test_y.shape[1]    
    score = np.zeros(Num_Cols)  
    
    for i in range(Num_Cols):
        score[i]=Models[i].score(DF_test_x,DF_test_y[[i]].squeeze())
        
    return score
    
def fitModels(clf, DF_train_x,DF_train_y):
    '''
    fits models to each target variable and returns a list of the fitted models
    
    Args:
    clf: the classifier instance to use
    DF_train_x: training feature dataframe
    DF_train_y: training target dataframe
    
    Returns:
    fit_models: a list of the fitted models for each target variable
    '''
    
    Num_Cols = DF_train_y.shape[1]
    fit_models = []

    
    for i in range(Num_Cols):
        clf.fit(DF_train_x,DF_train_y[[i]].squeeze())
        fit_models.append(clf)
    
    return fit_models

    
#svm


def main():
    filepath = "Data/Master_30_4.csv"
    # there are 835 assessors, each evaluated approximately 66 recordings
    
    #preparing the data
    DF_x,DF_y = splitXY(filepath)
    DF_x = DF_x.fillna(-1)    
    DF_y.isnull().sum() #only 5 missing values
    DF_y = TransformTarget(DF_y)  
    DF_train_x,DF_test_x,DF_train_y,DF_test_y = SplitTrainTest(DF_x,DF_y,train_percent = 0.75)
    
    #setting the hyperparameters to test
    nEstimators = [10,100,200,500,1000]  
    C = [10**x for x in range(-5,5)]
    
    #cross validation for different classifiers
    GBC_func = []
    for i in nEstimators:
        GBC_func.append(wrapper(GradientBoostingClassifier,n_estimators=i,learning_rate = 0.1))
    GBC_Score = CrossVal(DF_train_x,DF_train_y,GBC_func,k=3)
    
    RF_func = []    
    for i in nEstimators:
        RF_func.append(wrapper(RandomForestClassifier,n_estimators=i))
    RF_Score = CrossVal(DF_train_x,DF_train_y,RF_func,k=3)
    
    Logit_func = []
    for i in C:
        Logit_func.append(wrapper(LogisticRegression,C=i))
    Logit_Score = CrossVal(DF_train_x,DF_train_y,Logit_func,k=3)
    
    SVM_func = []
    for i in C:
        SVM_func.append(wrapper(SVC, C=i, kernel='poly', degree = 2))
    SVM_Score = CrossVal(DF_train_x,DF_train_y,SVM_func,k=3) 
    
    target_cols = list(DF_test_y.columns)
    
    #plots for the cross validation results
    Logit_score_avg = np.mean(Logit_Score, axis=1)
    position = 231    
    fig = plt.figure()
    for i in range(len(target_cols)):
        ax = fig.add_subplot(position)
        ax.plot(np.log(C),Logit_score_avg[i])
        ax.set_title(target_cols[i])
        ax.set_autoscaley_on(True)
        position += 1
    plt.tight_layout()
    fig.savefig("figures/logit_plots_5.5.15.png")
    
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
    fig.savefig("figures/rf_plots_5.5.15.png")
    
    GBC_score_avg = np.mean(GBC_Score, axis=1)
    position = 231    
    fig = plt.figure()
    for i in range(len(target_cols)):
        ax = fig.add_subplot(position)
        ax.plot(np.log(nEstimators),GBC_score_avg[i])
        ax.set_title(target_cols[i])
        ax.set_autoscaley_on(True)
        position += 1
    plt.tight_layout()
    fig.savefig("figures/GBC_plots_5.5.15.png")
    
    SVM_score_avg = np.mean(SVM_Score, axis=1)
    position = 231    
    fig = plt.figure()
    for i in range(len(target_cols)):
        ax = fig.add_subplot(position)
        ax.plot(np.log(nEstimators),SVM_score_avg[i])
        ax.set_title(target_cols[i])
        ax.set_autoscaley_on(True)
        position += 1
    plt.tight_layout()
    fig.savefig("figures/SVM_plots_5.5.15.png")
    
    
    #getting feature importance
    train_x, train_y,test_x, test_y = TrainTestClean(DF_train_x,DF_train_y,DF_test_x,DF_test_y)
    columns = train_x.columns.values.tolist()
    
    GBC_clf = GradientBoostingClassifier(n_estimators=500,learning_rate = 0.1)  
    GBC_clf.fit(train_x,train_y[[0]].squeeze())
    GBC_feature_importance= pd.DataFrame()
    GBC_feature_importance['features']=list(train_x.columns)
    GBC_feature_importance['importance'] = GBC_clf.feature_importances_
    GBC_feature_importance['importance'] = GBC_feature_importance['importance'] / max(GBC_feature_importance['importance'])
    GBC_feature_importance = GBC_feature_importance.sort('importance',ascending=False)
    
    RF_clf = RandomForestClassifier(n_estimators=500)
    RF_clf.fit(train_x,train_y[[0]].squeeze())
    RF_feature_importance = pd.DataFrame()
    RF_feature_importance['features'] = list(train_x.columns)
    RF_feature_importance['importance'] = RF_clf.feature_importances_
    RF_feature_importance['importance'] = RF_feature_importance['importance'] / max(RF_feature_importance['importance'])
    RF_feature_importance = RF_feature_importance.sort('importance',ascending=False)    
    
    feature_importance = pd.dataFrame()
    feature_importance['features'] = list(train_x.columns)[2:156]
    target_labels = train_y.columns.values.tolist()  
    
    #RF_models = fitModels(RF_clf,train_x[columns[2:156]],train_y)
    #for i in range(len(RF_models)):
     #   feature_importance[target_labels[i]] = RF_models[i].feature_importances_
      #  feature_importance[target_labels[i]] = feature_importance[target_labels[i]]/max(feature_importance[target_labels[i]])
    
    #RF_testScores = testScore(RF_models,test_x[columns[2:156]],test_y)
    
    #feature importance for logistic regression
    scores = np.zeros((len(columns),len(target_labels)))
    Logit_clf = LogisticRegression(C = 0.01)
    for i in range(len(target_labels)):  
        for j in range(len(columns)):
            cols = [col for col in train_x.columns if col not in [columns[j]]]
            Logit_clf.fit(train_x[cols],train_y[target_labels[i]].squeeze())
            temp_score = Logit_clf.score(test_x[cols],test_y[target_labels[i]].squeeze())
            print "target variable " + target_labels[i] + " without " + columns[j] + ": " + str(temp_score)
            scores[j][i] = temp_score
    
    #scores for logit with all features
    actual_scores = np.zeros(len(target_labels))
    for i in range(len(target_labels)):
        Logit_clf.fit(train_x,train_y[target_labels[i]].squeeze())
        actual_scores[i]=Logit_clf.score(test_x,test_y[target_labels[i]].squeeze())
    
    #testing without the time features
    unwanted_features = ['AssignmentDurationInSeconds', 'WorkTimeInSeconds']
    DF_train_x_2 = DF_train_x.drop(unwanted_features,axis=1)
    
    RF_func = []    
    for i in nEstimators:
        RF_func.append(wrapper(RandomForestClassifier,n_estimators=i))
    RF_Score_2 = CrossVal(DF_train_x_2,DF_train_y,RF_func,k=2)
    
    Logit_func = []
    for i in C:
        Logit_func.append(wrapper(LogisticRegression,C=i))
    Logit_Score_2 = CrossVal(DF_train_x_2,DF_train_y,Logit_func,k=2)

    #running the algorithms with only audio data, without audio data, with everything
    Audio_Features = columns[65:]
    GBC_audio_scores_avg = np.zeros((3,len(target_labels)))
    GBC = []
    GBC.append(GradientBoostingClassifier(n_estimators=500,learning_rate = 0.1))
    GBC_Scores_all = CrossVal(DF_train_x,DF_train_y,GBC,k=3)
    DF_train_x_audio = pd.concat([DF_train_x.ix[:,['WorkerId']],DF_train_x[Audio_Features]],axis=1,join_axes=[DF_train_x.index])
    GBC_Scores_audio = CrossVal(DF_train_x_audio,DF_train_y,GBC,k=3)
    GBC_Scores_noaudio = CrossVal(DF_train_x.drop(Audio_Features,axis=1),DF_train_y,GBC,k=3)
    GBC_audio_scores_avg[0] = np.mean(GBC_Scores_all, axis=1).ravel()
    GBC_audio_scores_avg[1] = np.mean(GBC_Scores_audio, axis=1).ravel()
    GBC_audio_scores_avg[2] = np.mean(GBC_Scores_noaudio, axis=1).ravel()
    GBC_labels = ["all","only_audio","no_audio"]
      
    fig = plt.figure()
    for i in range(GBC_audio_scores_avg.shape[0]):
        ax = fig.add_subplot(111)
        ax.plot(range(len(target_labels)),GBC_audio_scores_avg[i],label = GBC_labels[i])
        plt.xticks(range(len(target_labels)), target_labels, size="small")
        ax.set_autoscaley_on(True)
    plt.tight_layout()
    plt.legend(loc=4)
    fig.savefig("figures/GBC_audio_scores_5.5.15.png")
    
    GBC_fits = []
    GBC_feature_importance= pd.DataFrame()
    GBC_feature_importance['features']=train_x.columns.values.tolist()
    for i in range(len(target_labels)):    
        GBC_clf.fit(train_x,train_y[[i]].squeeze())
        GBC_fits.append(GBC_clf)
        col_name = target_labels[i] + " importance"
        GBC_feature_importance[col_name] = GBC_clf.feature_importances_
        GBC_feature_importance[col_name] = GBC_feature_importance[col_name] / max(GBC_feature_importance[col_name])
        print col_name + " completed"
        
    for i in range(len(target_labels)):
        col_name = target_labels[i] + " importance"
        GBC_feature_importance = GBC_feature_importance.sort(col_name,ascending=False)
        print "top features for " + col_name
        print GBC_feature_importance[[0,i+1]].head(20)
        print "bottom features for " + col_name
        print GBC_feature_importance[[0,i+1]].tail(20)
        
    GBC_feature_importance.to_csv("figures/GBC_feature_importance.csv")
        
        
    
    
    

