
# coding: utf-8

# In[2]:


# Death2GridSearch
# Author: Kevin Okiah
# 4/19/2019


# In[1]:


import yaml
import os
import pandas as pd
import numpy as np
from itertools import product
from pandas.tools.plotting import table
import  errno

#import Classifiers 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

#import Regressors


#Import model selection Utilities
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Import Evaluation metrics Classification
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# Import Evaluation metrics Regression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

#import Vizualization Libraries
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


# In[2]:


# Read YAML file - File with the classification Algorithms and hyperparameters
stream= open("setups/algos.yaml", 'r')
clf_dictionary = yaml.load(stream)


# In[14]:


#maps classfication objects to dict parms
clf_mapper = {MLPClassifier: "MLPClassifier",
              KNeighborsClassifier:"KNeighborsClassifier",
              AdaBoostClassifier:"AdaBoostClassifier",
              RandomForestClassifier:'RandomForestClassifier',
              LogisticRegression:'LogisticRegression',
              DecisionTreeClassifier: 'DecisionTreeClassifier',
              BaggingClassifier:'BaggingClassifier',
              ExtraTreesClassifier:'ExtraTreesClassifier',
              GradientBoostingClassifier:'GradientBoostingClassifier',
              ExtraTreeClassifier:'ExtraTreeClassifier'
             }

#BaggingClassifier, ExtraTreesClassifier,GradientBoostingClassifier, VotingClassifier, ExtraTreeClassifier


# In[15]:


als =list(clf_dictionary['Classification'].keys())


# In[16]:


#for i in als:
#    print(list(clf_dictionary['Classification'][i].keys()))


# In[17]:


#loading Dataset
#my_data = np.genfromtxt('data/numerai_data1.csv', delimiter=',',skip_header=1)
#dataX = pd.read_csv('data/heart.csv')


def ModelingData(X, y, n_folds):
    '''
    Function to stitch Features(X), Response Variable(y) 
    and n_folds for modeling
    X and y must ba arrays
    '''
    return (X, y, n_folds)

def  GenerateXandY(df, y_name):
    '''
    Function to generate X and y arrays from pandas dataframe
    
    df = pandas Dataframe
    y_name = Response varible or Target
    
    '''
    y = np.array(df[y_name])
    
    X = np.array(df.loc[:, df.columns != y_name].values)
    
    return(X, y)



# In[18]:


def run(a_clf, data, clf_hyper={}):
    '''
    This function takes in a classification object, dataset, and clf parms
    and performance metrics and runs
    '''
    M, L, n_folds = data # unpack data containter
    kf = StratifiedKFold(n_splits=n_folds) # Establish the cross validation
    ret = {} # classic explicaiton of results

    for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
        clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist
        clf.fit(M[train_index], L[train_index])
        pred = clf.predict(M[test_index])
        ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred),
               'precision': precision_score(L[test_index], pred),
               'recall': recall_score(L[test_index], pred),
               'f1_score': f1_score(L[test_index], pred),
               'roc_auc_score':roc_auc_score(L[test_index], pred)}
    return ret


# In[19]:


def Generate_Reports_Classification(Result, Algo, n, n_folds):
    '''
    Function generates reports for a given Classification Algorithm.
    '''
    
    try:
        temp_dir = "results/ClassifiersRuns/"
        #temp_dir_fig = temp_dir+"\\plots"
        os.makedirs(temp_dir)
        #os.makedirs(temp_dir_fig)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    folds =list(Result.keys())
    clfs =[]
    accuracy =[]
    precision =[]
    recall =[]
    f1_score =[]
    roc_auc_score =[]
    
    complete_name = os.path.join(temp_dir,Algo+"_runs_details.txt")


    f= open(complete_name,"a+")
    for i in folds:
        temp = Result[i]
        #print({temp['clf']})
        clfs =clfs+[temp['clf']]
        accuracy  = accuracy + [round(temp['accuracy'], 3)]
        precision = precision + [round(temp['precision'], 3)]
        recall= recall + [round(temp['recall'],3)]
        f1_score = f1_score + [round(temp['f1_score'],3)]
        roc_auc_score =roc_auc_score + [round(temp['roc_auc_score'],3)]
    metrics = [accuracy, precision , recall, f1_score, roc_auc_score ]
    metrics_names =['folds','accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']
    metrics_Names =['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']
    metrics_avg = [np.mean(accuracy), np.mean(precision) , np.mean(recall), np.mean(f1_score), np.mean(roc_auc_score)]

    metrics_avg = [ round(elem, 2) for elem in metrics_avg]

    #row folds data
    dat_dtype = {
    'names' : (metrics_names),
    'formats' : ('i', 'd', 'd', 'd', 'd', 'd')}
    dat = np.zeros(n_folds, dat_dtype)

    dat['folds'] = folds
    dat['accuracy'] = accuracy
    dat['precision'] = precision
    dat['recall'] = recall
    dat['f1_score'] = f1_score
    dat['roc_auc_score'] = roc_auc_score

    #averages
    dat_dtype2 = {
    'names' : (metrics_names),
    'formats' : ('d', 'd', 'd', 'd', 'd', 'd')}
    dat2 = np.zeros(1, dat_dtype2)

    dat2['folds'] = n_folds
    dat2['accuracy'] = metrics_avg[0]
    dat2['precision'] = metrics_avg[1]
    dat2['recall'] = metrics_avg[2]
    dat2['f1_score'] = metrics_avg[3]
    dat2['roc_auc_score'] = metrics_avg[4]
    
    f.write('-------------------------------------------------------------------' + '\n')
    f.write('Param Set ' + str(n) + '\n')
    f.write('-------------------------------------------------------------------' + '\n')

    x = PrettyTable(dat.dtype.names)
    for row in dat:
        x.add_row(row)

    f.write(str(x))
    f.write('\n')

    f.write('-------------------------------------------------------------------' + '\n')
    f.write('Average Scores for folds' + '\n')
    f.write('-------------------------------------------------------------------' + '\n')

    y = PrettyTable(dat2.dtype.names)
    for row in dat2:
        y.add_row(row)

    f.write(str(y))

    f.write('\n')


    f.write(str(clfs[0]) + '\n')

    f.close()

    #Generate_Bars(folds, Algo,n, metrics, metrics_Names)
    

    return(folds, Algo, n, metrics, metrics_names, metrics_Names, metrics_avg, clfs[0])


# In[25]:


def SummaryReport(file):
    
    '''
    Function to generate a summary report.
    
    '''

    try:
        temp_dir_fig = "results/Summary"
        os.makedirs(temp_dir_fig)

    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    metrics_Names =['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']
    temp =file['Summary'].keys()
    Final_data = []
    X_final ={}
    Algos_List =list(temp)
    for al in range(len(Algos_List)):
        #keys = list(file['Summary'][Algos_List[al]].keys())
        temp_data = []
        summary_data = []
        #Final_data = []
        for i in range(5): # number of accruacy measures
            temp_data = []
            for k, v in file['Summary'][Algos_List[al]].items():
                temp_data = temp_data+[(file['Summary'][Algos_List[al]][k][i])]
            summary_data =summary_data+ [temp_data]
        Final_data =Final_data+ [summary_data]

    for met in range(len(metrics_Names)):
        filename = str(metrics_Names[met])+"_Summary.jpg"
        fig_name = os.path.join(temp_dir_fig,filename)
        Met_score =[]
        for i in Final_data:
            Met_score = Met_score +[i[met]]
        X = dict(zip(Algos_List, Met_score))
        X_final.update({metrics_Names[met]:X})
        plt.figure(figsize=(15,5))
        plt.boxplot(X.values(), labels=X.keys())
        #sns.boxplot(x=list(X.values()), y=list(X.keys()))
        plt.title(metrics_Names[met], fontsize=18)
        plt.xlabel('Algorithm', fontsize=10)
        plt.xticks(rotation=90)
        plt.ylabel(metrics_Names[met], fontsize=10)
        plt.savefig(fig_name)

    try:
        temp_dir = "results/Summary"
        #temp_dir_fig = temp_dir+"\\plots"
        os.makedirs(temp_dir)
        #os.makedirs(temp_dir_fig)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    bestscore =BestModel(file, X_final)
    summary = SummaryTable(X_final)
    summary_name = os.path.join(temp_dir,"Summary.txt")


    f= open(summary_name,"a+")

    f.write("------------------------------------------------------------------------------------"+'\n')
    f.write("                                Performance Summary                                 "+'\n')
    f.write("------------------------------------------------------------------------------------"+'\n')
    f.write("                       Best performing model by perfomance Metrics                  "+'\n')
    f.write("------------------------------------------------------------------------------------"+'\n')
    f.write(str(bestscore))
    f.write('\n')
    f.write('\n')
    f.write('\n')

    f.write("------------------------------------------------------------------------------------"+'\n')
    f.write("                       Summary statistics by model                                  "+'\n')
    f.write("------------------------------------------------------------------------------------"+'\n')
    f.write(str(summary))
    f.write("------------------------------------------------------------------------------------"+'\n')
    f.write('\n')


    f.close()
    #plt.show()
    #print("Processing complete.....")
    return(X_final)


# In[26]:


def SummaryTable(dat):
    '''
    Generates summary table
    '''

    from prettytable import PrettyTable
    title = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']
    models = list(dat[title[0]].keys())

    scores_final = {}
    for met in title:
        scores ={}
        n =0
        for i in list(dat[met].values()):
            AVG = round(np.mean(i), 3)
            MAX = round(np.max(i), 3)
            MIN = round(np.min(i), 3)
            StdDEV = round(np.std(i), 3)
            temp = [AVG, MAX, MIN, StdDEV]
            scores.update({models[n]:{' [AVG, MAX, MIN, StdDEV]':temp}})
            n =n+1
        #print(scores)

        scores_final.update({met:scores})
    x = PrettyTable()
    x.field_names = ["Measure", "Model", "AVG", "MAX", "MIN", "StdDEV"]
    for k , v in scores_final.items():
        #print(k)
        for k2, v2 in v.items():
            #print(k2)
            for k3, v3 in v2.items():
                #print([k, k2, v3])
                x.add_row([k, k2, v3[0], v3[1], v3[2], v3[3]])
    return(x)

def BestModel(file, dat):
    '''
    Function to return the best model and hyper params by different measures
    '''
    hypers = []
    for key, val in file['clfs'].items():
        for key1, val1 in val.items():
            hypers = hypers+[val1]

    models =[]
    indexs = []
    for key1, val1 in file['Summary'].items():
        for key3, val3 in val1.items():
                models =models +[key1]
                indexs = indexs+ [key3]
    scores_track = []

    for k,j in dat.items():
        temp = []
        #print(k)
        for k1, j1 in j.items():
            temp = temp + j1
        scores_track = scores_track+ [temp]

    title = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score']

    x = PrettyTable()
    x.field_names = ["Measure", "Model", "BestScore", "Hyper Paramaters"]

    for i in range(len(title)):
        tempX = [title[i],models[scores_track[i].index(max(scores_track[i]))],scores_track[i][scores_track[i].index(max(scores_track[i]))], hypers[scores_track[i].index(max(scores_track[i]))] ]
        x.add_row(tempX)
        #print("-----------------------------------")
    return(x)


# In[27]:


def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


# In[28]:


dataX = pd.DataFrame()

def main(clf_dict = clf_dictionary, mapper = clf_mapper, data = dataX, Response ='target', n_folds = 10 , Analysis ='Classification'):
    
    '''
    Function to put everything together
    
    clf_dict - Dictionary of Algorithms with hyperparamaters set from the yaml file
    mapper= Maps clf dictionary to sklearn learn classifer objects
    data = Cleaned pandas Dataframe ready for modeling
    Resporse = this is the y
    
    '''   
    #generate X and y from pandas df
    X, y = GenerateXandY(data,Response)

    #stitch X, y and n_foldes together
    data =ModelingData(X, y, n_folds)

    Metrics_Summary ={} #metrics summary
    Clfs_Summary ={} #clfs Summary
    Super_Dictionary ={} #super_disctionary

    for key1 , value1 in clf_dict.items():
        if key1 ==Analysis:
            # tracks param combination
            for key2 , value2 in value1.items():
                t =0
                print("Running....", key2)
                clf = list(clf_mapper.keys())[list(clf_mapper.values()).index(key2)]
                #print(clf)
                Algo_avg = {}
                Clfs_Set = {}
                try:
                    key3,value3 =zip(*value2.items())
                    for values in product(*value3):
                        hyperset =dict(zip(key3, values))
                        #print(hyperset)
                        result = run(clf, data, hyperset)
                        folds_, Algo_, n_, metrics_, metrics_names_, metrics_Names_, metrics_avg_, clfs_= Generate_Reports_Classification(result, key2, t, n_folds) #generate reports and plots
                        Clfs_Set.update({t:hyperset})
                        Algo_avg.update({t:metrics_avg_})
                        t = t + 1
                except AttributeError:
                    print("missing keys and values")
                Metrics_Summary.update({key2:Algo_avg})
                Clfs_Summary.update({key2:Clfs_Set})
                Super_Dictionary.update({"Summary":Metrics_Summary, "clfs":Clfs_Summary})
        dat = SummaryReport(Super_Dictionary)
        BestModel(Super_Dictionary, dat)
        
        datatable = dat.copy()
        algo_list = list(datatable['accuracy'])
        metrics =list(datatable.keys())
        
        results_temp =pd.DataFrame()
        temp =[]
        for j in metrics:
            temp =[]
            for i in algo_list:
                #print(i, ":",round(float(np.average(datatable[j][i])), 2))
                temp = temp+ [round(float(np.max(datatable[j][i])), 2)]
            results_temp[j] = temp
            results_temp.index = algo_list

        s = results_temp.style.apply(highlight_max)
        print("processing Completed Successfully")
    return(s)


# In[29]:


# run main 
#main(data =dataX, Response='target', n_folds=5)

