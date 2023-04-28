############################################################################################
##EIGHTH SCRIPT - PSEUDO PARTIAL EFFECTS FOR POSITIVE AND NEGATIVE AFFECTS ON EXTENDED SET##
############################################################################################

import re 
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(1123581321)

'''
COMMENTS

This is the eighth script regarding the Extended Set 
producing the results in "Machine Learning in the 
Prediction of Human Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., and et al.

Aim of this script is to compute the pseudo partial effects
for positive and negative affects.

Id est, we identify which variables are positively related with
positive and negative affects. and which aren't.
'''

def linreg_train_test(X_train, y_train, X_test, y_test):
    
    lineareg = LinearRegression()
    
    X_const_train = sm.add_constant(X_train, has_constant = 'add')
    
    X_const_test = sm.add_constant(X_test, has_constant = 'add')
    
    lineareg_fitted = lineareg.fit(X_const_train, y_train)
    
    lineareg_yhat_test = lineareg_fitted.predict(X_const_test)

    Mse_lineareg_test = ((lineareg_yhat_test - y_test)**2).mean()
    
    lineareg_yhat_train = lineareg_fitted.predict(X_const_train)

    Mse_lineareg_train = ((lineareg_yhat_train - y_train)**2).mean()  
    
    lineareg_yhat_train_round = np.round(lineareg_yhat_train)
        
    Test_R2 = r2_score(y_test, lineareg_yhat_test)
    
    Train_R2 = r2_score(y_train, lineareg_yhat_train)

    list_of_results = [Mse_lineareg_test, Mse_lineareg_train, Test_R2, Train_R2, lineareg_fitted]
    
    return list_of_results

def RandomForest(X_train, y_train, if_bootstrap,
                 optim, n_trees, n_max_feats, 
                 n_max_depth, n_min_sample_leaf, 
                 n_cv, X_test = None,
                 y_test = None):
        
    if optim == True:
        
        rf = RandomForestRegressor(bootstrap = if_bootstrap)

        pruning_dict = {'n_estimators':n_trees,
                'max_features': n_max_feats,
                'max_depth':n_max_depth,
                'min_samples_leaf':n_min_sample_leaf
                }
        
        rf_regr_optim = GridSearchCV(rf, 
                        pruning_dict, 
                        cv = n_cv, 
                        n_jobs = -1,
                        scoring = 'neg_mean_squared_error')
        
    else:
        
        rf_regr_optim = RandomForestRegressor(n_estimators = n_trees[0],
                                              max_features = n_max_feats[0],
                                              max_depth = n_max_depth[0])
        
    rf_regr_fitted = rf_regr_optim.fit(X_train, y_train)
        
    best_rf = rf_regr_fitted.best_estimator_
    
    yhat_train = best_rf.predict(X_train)

    Train_MSE = ((yhat_train - y_train)**2).mean() 
    
    results_from_cv = rf_regr_fitted.cv_results_
    
    if X_test is None and y_test is None:
        
        print('No out of sample accuracy was computed')
    
    else:
        
        yhat_test = best_rf.predict(X_test)

        Test_MSE = ((yhat_test - y_test)**2).mean() 
        
        Train_R2 = r2_score(y_train, yhat_train)
        
        Test_R2 = r2_score(y_test, yhat_test)
        
    list_of_results = [rf_regr_fitted, best_rf, results_from_cv, Test_MSE, Train_MSE, Test_R2, Train_R2]
    
    return list_of_results

def GradBoostReg(X_train, y_train, 
                 lr,
                 n_iters,  
                 max_depth,  
                 subsample_frac,
                 max_feats,
                 n_cv, 
                 X_test = None,
                 y_test = None):
        
    gb = GradientBoostingRegressor(verbose = 1)

    optim_dict = {'n_estimators': n_iters,
                  'learning_rate': lr,
                  'max_depth': max_depth,
                  'subsample': subsample_frac,
                  'max_features': max_feats}
        
    gb_regr_optim = GridSearchCV(gb, 
                                 optim_dict, 
                                 cv = n_cv, 
                                 n_jobs = -1,  
                                 scoring = 'neg_mean_squared_error')

    gb_regr_fitted = gb_regr_optim.fit(X_train, y_train)
        
    best_gb = gb_regr_fitted.best_estimator_
    
    yhat_train = best_gb.predict(X_train)

    Train_MSE = ((yhat_train - y_train)**2).mean() 
    
    results_from_cv = gb_regr_fitted.cv_results_
    
    if X_test is None and y_test is None:
        
        print('No out of sample accuracy was computed')
    
    else:
        
        yhat_test = best_gb.predict(X_test)

        Test_MSE = ((yhat_test - y_test)**2).mean() 
        
        Train_R2 = r2_score(y_train, yhat_train)
        
        Test_R2 = r2_score(y_test, yhat_test)

    list_of_results = [gb_regr_fitted, best_gb, results_from_cv, Test_MSE, Train_MSE, Test_R2, Train_R2]
    
    return list_of_results

read_path = 'C:\\Users\\niccolo.gentile\\Desktop\\Joined_paper\\New_cleared_codes_to_share_230323\\New_KS_codes\\New_scripts_and_results\\New_dsets_17042023\\'

X_train_plks = pd.read_csv(read_path + 'X_train_plks_stand.csv') 

X_test_plks = pd.read_csv(read_path + 'X_test_plks_stand.csv')

#Differently from when importing the whole Extended set,
#in this case a "Unnamed: 0" column with the row indexes is 
#created, hence we drop it.

X_train_plks.drop(["Unnamed: 0"], axis = 1, inplace = True)

X_test_plks.drop(["Unnamed: 0"], axis = 1, inplace = True)

#We also need to import the (Full) Extended train and 
#test sets as in those there is also life satisfaction and the
#pids, which will be useful.

train_ks = pd.read_csv(read_path + 'train_ks_stand.csv') 

test_ks = pd.read_csv(read_path + 'test_ks_stand.csv')

pids_train_plks = train_ks['pid']

pids_test_plks = test_ks['pid']

del train_ks, test_ks

X_train_plks["pid"] = pids_train_plks

X_test_plks["pid"] = pids_test_plks

del pids_train_plks, pids_test_plks

################################
###IMPORTING AFFECT VARIABLES###
################################

bdp_path = 'C:\\Some\\Local\\Path\\'

vars_bdp = ['syear', 'pid', 'bdp0201', 'bdp0202', 'bdp0203', 'bdp0204']

bdp_obj = pyreadstat.read_dta(bdp_path + 'bdp.dta', usecols = vars_bdp)

bdp_data = bdp_obj[0]

bdp_data.shape

#Notice that:

#(30956, 6)

#bdp_data_13 = bdp_data[bdp_data['syear'] == 2013]

#bdp_data_13.shape

#(30956, 6)

#(bdp_data_13 != bdp_data).sum().sum()

#0

#Since bdp_data['syear'].unique()

#array([2013], dtype=int64)

#Missing and negative values in bdp?

for i in list(bdp_data):
    
    print([i, "Negatives in " + i + ": " + str(np.sum(bdp_data[i] < 0)), "NaNs in " + i + ": " + str(bdp_data[i].isna().sum())])
    

#['pid', 'Negatives in pid: 0', 'NaNs in pid: 0']
#['syear', 'Negatives in syear: 0', 'NaNs in syear: 0']
#['bdp0201', 'Negatives in bdp0201: 5037', 'NaNs in bdp0201: 0']
#['bdp0202', 'Negatives in bdp0202: 5061', 'NaNs in bdp0202: 0']
#['bdp0203', 'Negatives in bdp0203: 5047', 'NaNs in bdp0203: 0']
#['bdp0204', 'Negatives in bdp0204: 5035', 'NaNs in bdp0204: 0']

#and since we know that negative values are a different way
#to labeel missingness

bdp_data_nomiss = bdp_data[(bdp_data >= 0).all(1)]

#bdp_data_nomiss.shape

#(25861, 6)

X_train_plks_bdp = pd.merge(X_train_plks, bdp_data_nomiss, on = 'pid', how = 'inner')

X_test_plks_bdp = pd.merge(X_test_plks, bdp_data_nomiss, on = 'pid', how = 'inner')

const_in_test = []

for i in list(X_test_plks_bdp):
        
    if X_test_plks_bdp[i].nunique() == 1:
        
        const_in_test.append(i)
            
        X_train_plks_bdp.drop(i, axis = 1, inplace = True)
            
        X_test_plks_bdp.drop(i, axis = 1, inplace = True)
        
len(const_in_test)

const_in_train = []

for i in list(X_train_plks_bdp):
        
    if X_train_plks_bdp[i].nunique() == 1:
        
        const_in_train.append(i)
            
        X_train_plks_bdp.drop(i, axis = 1, inplace = True)
            
        X_test_plks_bdp.drop(i, axis = 1, inplace = True)
        
len(const_in_train)

###################
##NEGATIVE AFFECT##
###################

X_train_plks_bdp['neg_affect'] = 1/3 * X_train_plks_bdp['bdp0201'] + 1/3 * X_train_plks_bdp['bdp0202'] +  1/3 * X_train_plks_bdp['bdp0204']

X_test_plks_bdp['neg_affect'] = 1/3 * X_test_plks_bdp['bdp0201'] + 1/3 * X_test_plks_bdp['bdp0202'] +  1/3 * X_test_plks_bdp['bdp0204']

#####################
##LINEAR REGRESSION##
#####################

X_train_plks_bdp_neg = X_train_plks_bdp.drop(['neg_affect', 'bdp0201', 'bdp0202', 'bdp0204', 'bdp0203', 'pid'], axis = 1)

y_train_plks_bdp_neg = X_train_plks_bdp['neg_affect']

X_test_plks_bdp_neg = X_test_plks_bdp.drop(['neg_affect', 'bdp0201', 'bdp0202', 'bdp0204', 'bdp0203', 'pid'], axis = 1)

y_test_plks_bdp_neg = X_test_plks_bdp['neg_affect']

linreg_plks_bdp_neg = linreg_train_test(X_train = X_train_plks_bdp_neg, 
                                        y_train = y_train_plks_bdp_neg, 
                                        X_test = X_test_plks_bdp_neg, 
                                        y_test = y_test_plks_bdp_neg)

##################
###RANDOM FOREST##
##################

start_time = time.time()

RF_plksbdp_neg = RandomForest(X_train = X_train_plks_bdp_neg, 
                              y_train = y_train_plks_bdp_neg, 
                              if_bootstrap = True,
                              optim = True, 
                              n_trees = [1000], 
                              n_max_feats = [60], 
                              n_max_depth = [60], 
                              n_min_sample_leaf = [1], 
                              n_cv = 4, 
                              X_test = X_test_plks_bdp_neg,
                              y_test = y_test_plks_bdp_neg)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

######################
###GRADIENT BOOSTING##
######################

start_time = time.time()

GB_plksbdp_neg = GradBoostReg(X_train = X_train_plks_bdp_neg, 
                              y_train = y_train_plks_bdp_neg, 
                              lr = [0.01],
                              n_iters = [2000],
                              max_depth = [6], 
                              subsample_frac = [0.75],
                              max_feats = [25], 
                              n_cv = 4, 
                              X_test = X_test_plks_bdp_neg,
                              y_test = y_test_plks_bdp_neg)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Attempts

###################
##POSITIVE AFFECT##
###################

#Positive affect is simply bdp0203.

#####################
##LINEAR REGRESSION##
#####################

X_train_plks_bdp_pos = X_train_plks_bdp.drop(['bdp0201', 'bdp0202', 'bdp0204', 'bdp0203', 'pid', 'neg_affect'], axis = 1)

y_train_plks_bdp_pos = X_train_plks_bdp['bdp0203']

X_test_plks_bdp_pos = X_test_plks_bdp.drop(['bdp0201', 'bdp0202', 'bdp0204', 'bdp0203', 'pid', 'neg_affect'], axis = 1)

y_test_plks_bdp_pos = X_test_plks_bdp['bdp0203']

linreg_plks_bdp_pos = linreg_train_test(X_train = X_train_plks_bdp_pos, 
                                        y_train = y_train_plks_bdp_pos, 
                                        X_test = X_test_plks_bdp_pos, 
                                        y_test = y_test_plks_bdp_pos)

##################
###RANDOM FOREST##
##################

start_time = time.time()

RF_plksbdp_pos = RandomForest(X_train = X_train_plks_bdp_pos, 
                              y_train = y_train_plks_bdp_pos, 
                              if_bootstrap = True,
                              optim = True, 
                              n_trees = [1000], 
                              n_max_feats = [70], 
                              n_max_depth = [60], 
                              n_min_sample_leaf = [1],
                              n_cv = 4, 
                              X_test = X_test_plks_bdp_pos,
                              y_test = y_test_plks_bdp_pos)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

######################
###GRADIENT BOOSTING##
######################

start_time = time.time()

GB_plksbdp_pos = GradBoostReg(X_train = X_train_plks_bdp_pos, 
                              y_train = y_train_plks_bdp_pos, 
                              lr = [0.005],
                              n_iters = [2500],
                              max_depth = [9], 
                              subsample_frac = [0.75],
                              max_feats = [25], 
                              n_cv = 4, 
                              X_test = X_test_plks_bdp_pos,
                              y_test = y_test_plks_bdp_pos)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

###################################
###PARTIAL EFFECTS' COMPUTATION####
###################################

####################################
####LINEAR REGRESSION - NEGATIVE####
####################################

varlist = pd.read_csv(path + 'PI_neg_OLS_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() #only keep the list of variables

differences_list = []

for i in varlist:  
            
    tmp = X_test_plks_bdp_neg.copy() 
    
    tmp_const = sm.add_constant(tmp, has_constant = 'add')

    if tmp_const[i].nunique() == 2: 
        
        print('# ' + i + ' is binary')
              
        stub = re.split("_", i)[0]
      
        print(stub)
                
        fullvarlist = pd.read_csv(path + 'PI_neg_OLS_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() 
                
        tomanipulate = []
        
        for j in fullvarlist:
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
        
        for j in tomanipulate:
            
            tmp_const[j] = tmp_const[j].min()                     
                
        del tomanipulate         
        
        low = tmp_const[i].min() 
        
        high = tmp_const[i].max() 
        
    else:    
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp_const[i].describe().loc['25%'] 
        
        high = tmp_const[i].describe().loc['75%'] 
      
    tmp_const[i] = low
    
    avg_yhat_low = linreg_plks_bdp_neg[-1].predict(tmp_const).mean()
        
    tmp_const[i] = high
               
    avg_yhat_high = linreg_plks_bdp_neg[-1].predict(tmp_const).mean()
    
    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp_const
 
PI_OLS_plks_sorted = pd.read_csv(path + 'PI_neg_OLS_df.csv')

not_computed = ['not computed'] * (PI_OLS_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_OLS_plks_sorted['Pseudo effect'] = differences_list

PI_OLS_plks_sorted.to_csv(path + 'PI_neg_OLS_df.csv', index = False)

############################
##RANDOM FOREST - NEGATIVE##
############################

varlist = pd.read_csv(path + 'PI_neg_RF_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() 

differences_list = []

for i in varlist:  
        
    tmp = X_test_plks_bdp_neg.copy() 
  
    if tmp[i].nunique() == 2: 
        
        print('# ' + i + ' is binary')
                
        stub = re.split("_", i)[0]
      
        print(stub)
                
        fullvarlist = pd.read_csv(path + 'PI_neg_RF_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() 
        
        tomanipulate = []
        
        for j in fullvarlist:
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
        
        for j in tomanipulate:
            
            tmp[j] = tmp[j].min() 
        
        del tomanipulate         
        
        low = tmp[i].min() 
        
        high = tmp[i].max() 
        
        
    else:        
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp[i].describe().loc['25%'] 
        
        high = tmp[i].describe().loc['75%'] 
        
    tmp[i] = low
                
    avg_yhat_low = RF_plksbdp_neg[1].predict(tmp).mean()
            
    tmp[i] = high
               
    avg_yhat_high = RF_plksbdp_neg[1].predict(tmp).mean()
    
    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp
 
PI_RF_plks_sorted = pd.read_csv(path + 'PI_neg_RF_df.csv')

not_computed = ['not computed'] * (PI_RF_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_RF_plks_sorted['Pseudo effect'] = differences_list

PI_RF_plks_sorted.to_csv(path + 'PI_neg_RF_df.csv', index = False)

################################
##GRADIENT BOOSTING - NEGATIVE##
################################

varlist = pd.read_csv(path + 'PI_neg_GB_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() 

differences_list = []

for i in varlist:  
        
    tmp = X_test_plks_bdp_neg.copy() 
  
    if tmp[i].nunique() == 2: 
        
        print('# ' + i + ' is binary')
                
        stub = re.split("_", i)[0]
      
        print(stub)
                
        fullvarlist = pd.read_csv(path + 'PI_neg_GB_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() 
        
        tomanipulate = []
        
        for j in fullvarlist:
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
        
        for j in tomanipulate:
            
            tmp[j] = tmp[j].min() 
        
        del tomanipulate         
        
        low = tmp[i].min() 
        
        high = tmp[i].max() 
        
        
    else:        
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp[i].describe().loc['25%'] 
        
        high = tmp[i].describe().loc['75%'] 
        
    tmp[i] = low
                
    avg_yhat_low = GB_plksbdp_neg[1].predict(tmp).mean()
            
    tmp[i] = high
               
    avg_yhat_high = GB_plksbdp_neg[1].predict(tmp).mean()
    
    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp
 
PI_GB_plks_sorted = pd.read_csv(path + 'PI_neg_GB_df.csv')

not_computed = ['not computed'] * (PI_GB_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_GB_plks_sorted['Pseudo effect'] = differences_list

PI_GB_plks_sorted.to_csv(path + 'PI_neg_GB_df.csv', index = False)

####################################
####LINEAR REGRESSION - POSITIVE####
####################################

varlist = pd.read_csv(path + 'PI_pos_OLS_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() 

differences_list = []

for i in varlist:  
            
    tmp = X_test_plks_bdp_pos.copy() 
    
    tmp_const = sm.add_constant(tmp, has_constant = 'add')

    if tmp_const[i].nunique() == 2: 
        
        print('# ' + i + ' is binary')
              
        stub = re.split("_", i)[0]
      
        print(stub)
                
        fullvarlist = pd.read_csv(path + 'PI_pos_OLS_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() 
                
        tomanipulate = []
        
        for j in fullvarlist:
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
        
        for j in tomanipulate:
            
            tmp_const[j] = tmp_const[j].min()                     
                
        del tomanipulate         
        
        low = tmp_const[i].min() 
        
        high = tmp_const[i].max() 
        
    else:  
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp_const[i].describe().loc['25%'] 
        
        high = tmp_const[i].describe().loc['75%'] 
      
    tmp_const[i] = low
    
    avg_yhat_low = linreg_plks_bdp_pos[-1].predict(tmp_const).mean()
        
    tmp_const[i] = high
               
    avg_yhat_high = linreg_plks_bdp_pos[-1].predict(tmp_const).mean()
    
    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp_const
 
PI_OLS_plks_sorted = pd.read_csv(path + 'PI_pos_OLS_df.csv')

not_computed = ['not computed'] * (PI_OLS_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_OLS_plks_sorted['Pseudo effect'] = differences_list

PI_OLS_plks_sorted.to_csv(path + 'PI_pos_OLS_df.csv', index = False)

############################
##RANDOM FOREST - POSITIVE##
############################

varlist = pd.read_csv(path + 'PI_pos_RF_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() 

differences_list = []

for i in varlist:  
        
    tmp = X_test_plks_bdp_pos.copy() 
  
    if tmp[i].nunique() == 2: 
        
        print('# ' + i + ' is binary')
                
        stub = re.split("_", i)[0]
      
        print(stub)
                
        fullvarlist = pd.read_csv(path + 'PI_pos_RF_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() 
        
        tomanipulate = []
        
        for j in fullvarlist:
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
        
        for j in tomanipulate:
            
            tmp[j] = tmp[j].min() 
        
        del tomanipulate         
        
        low = tmp[i].min() 
        
        high = tmp[i].max() 
        
        
    else:        
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp[i].describe().loc['25%'] 
        
        high = tmp[i].describe().loc['75%'] 
        
    tmp[i] = low
                
    avg_yhat_low = RF_plksbdp_pos[1].predict(tmp).mean()
            
    tmp[i] = high
               
    avg_yhat_high = RF_plksbdp_pos[1].predict(tmp).mean()
    
    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp
 
PI_RF_plks_sorted = pd.read_csv(path + 'PI_pos_RF_df.csv')

not_computed = ['not computed'] * (PI_RF_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_RF_plks_sorted['Pseudo effect'] = differences_list

PI_RF_plks_sorted.to_csv(path + 'PI_pos_RF_df.csv', index = False)

################################
##GRADIENT BOOSTING - POSITIVE##
################################

varlist = pd.read_csv(path + 'PI_pos_GB_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() 

differences_list = []

for i in varlist:  
        
    tmp = X_test_plks_bdp_pos.copy() 
  
    if tmp[i].nunique() == 2:
        
        print('# ' + i + ' is binary')
                
        stub = re.split("_", i)[0]
      
        print(stub)
                
        fullvarlist = pd.read_csv(path + 'PI_pos_GB_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() 
        
        tomanipulate = []
        
        for j in fullvarlist:
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
        
        for j in tomanipulate:
            
            tmp[j] = tmp[j].min() 
        
        del tomanipulate         
        
        low = tmp[i].min() 
        
        high = tmp[i].max() 
        
        
    else:        
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp[i].describe().loc['25%'] 
        
        high = tmp[i].describe().loc['75%'] 
        
    tmp[i] = low
                
    avg_yhat_low = GB_plksbdp_pos[1].predict(tmp).mean()
            
    tmp[i] = high
               
    avg_yhat_high = GB_plksbdp_pos[1].predict(tmp).mean()
    
    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp
 
PI_GB_plks_sorted = pd.read_csv(path + 'PI_pos_GB_df.csv')

not_computed = ['not computed'] * (PI_GB_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_GB_plks_sorted['Pseudo effect'] = differences_list

PI_GB_plks_sorted.to_csv(path + 'PI_pos_GB_df.csv', index = False)



