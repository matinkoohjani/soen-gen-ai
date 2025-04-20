from pycaret.classification import *
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle


col_list = ['AvgCountLineCode',
 'SumCyclomatic',
 'keywords',
 'if_else_while_operators',
 'MaxNesting',
 'CountLineBlank',
 'CountDeclFunction',
 'CountLineCodeDecl']

tuned_model_list_datas = defaultdict()
i = 0

for folder_name in os.listdir(''):
    print(folder_name)
    file_name = [x for x in os.listdir('') if 'train' in x][0]
    data = pd.read_csv('', index_col=0)
    data = data[col_list+['actual label']]
    reg1 = setup(data = data, target = 'actual label', index=False, session_id=42)
    # Set model to tune
    model_list = [MLPClassifier()]
    tuned_model_list = []

    for model in model_list:
        tuned_model = tune_model(model)
        tuned_model.set_params(random_state=42)
        tuned_model_list.append(tuned_model)
        print(tuned_model, '\n')
    
    tuned_model_list_datas[folder_name] = tuned_model_list
    i += 1


with open('', 'wb') as f:
    pickle.dump(tuned_model_list_datas, f)