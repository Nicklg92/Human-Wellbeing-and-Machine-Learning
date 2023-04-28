####################################################
##FOURTH SCRIPT - POST-LASSO EXTENDED SET CREATION##
####################################################

import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV

np.random.seed(1123581321)

'''
COMMENTS

This is the fourth script regarding the Extended Set 
producing the results in "Machine Learning in the 
Prediction of Human Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., and et al.

Aim of this script is to create the Post-LASSO Extended set.
'''

read_path = 'C:\\Some\\Local\\Path\\'

train_ks = pd.read_csv(read_path + 'train_ks_stand.csv') 

test_ks = pd.read_csv(read_path + 'test_ks_stand.csv')

y_train_ks = train_ks['lsat']

y_test_ks = test_ks['lsat']

X_train_ks = train_ks.drop(['lsat', 'pid'], axis = 1)

X_test_ks = test_ks.drop(['lsat', 'pid'], axis = 1)
        
const_in_test = []

for i in list(X_test_ks):
        
    if X_test_ks[i].nunique() == 1:
        
        const_in_test.append(i)
            
        X_train_ks.drop(i, axis = 1, inplace = True)
            
        X_test_ks.drop(i, axis = 1, inplace = True)
        
len(const_in_test)

#10

#X_train_ks.shape

#23454 x 542

#X_test_ks.shape

#5864 x 542

#We ex-ante delete the columns that we already know are
#collinear (LASSO would drop them anyway).

multicoll = ["i11112", "m11101", "m11122", "m11123", 
             "plb0097_-2.0", "hlf0011_h_nan", "ple0004_2.0",
             "ple0164_-2.0", "plb0040_nan", "plj0022_-2.0",
             "e11103_2.0", "plb0022_h_5.0", "plb0022_h_9.0",
             "plb0035_nan", "plj0116_h_-2.0", "e11106_nan",
             "plb0041_-2.0", "hlf0073_-2.0", "plb0031_h_-2.0",
             "hlf0092_h_nan", "plb0103_-2.0", "plb0156_v1_nan"]

X_train_ks.drop(multicoll, axis = 1, inplace = True)

X_test_ks.drop(multicoll, axis = 1, inplace = True)

###################
###RUNNING LASSO###
###################

LASSO_t = []

#default tol = 0.0001

LASSO_t.append(LassoCV(cv = 4, n_jobs= - 1).fit(X_train_ks, y_train_ks))

LASSO_t.append(LASSO_t[0].score(X_test_ks, y_test_ks)) #Test R-squared

LASSO_t.append(((LASSO_t[0].predict(X_test_ks) - y_test_ks)**2).mean()) #Test MSE

LASSO_t.append(LASSO_t[0].score(X_train_ks, y_train_ks)) #Train R-squared

LASSO_t.append(((LASSO_t[0].predict(X_train_ks) - y_train_ks)**2).mean()) #Train MSE

#What is the optimal lambda? 

LASSO_t.append(LASSO_t[0].alpha_)

#Out of which grid?

LASSO_t.append(LASSO_t[0].alphas_)

#And what are the coefficients?

LASSO_t.append(LASSO_t[0].coef_)

#LASSO's Test R2

LASSO_t[1]

#0.29

#LASSO's Test MSE

LASSO_t[2]

#2.20

#LASSO's Train R2

LASSO_t[3]

#0.30

#LASSO's Train MSE

LASSO_t[4]

#2.14

#LASSO's optimal lambda (alpha)

LASSO_t[5]

#0.01

#Out of which grid?

LASSO_t[6].max()

LASSO_t[6].min()

#0.46, 0.00046

#Where LASSO_t[6][0] is the max, and LASSO_t[6][99] the smallest. Where does 
#our optimal lambda stand?

list(LASSO_t[6]).index(LASSO_t[5])

#55th. Let's investigate the coefficients now: how many shrunk to 0?

np.sum(LASSO_t[7] == 0)

#295, implying 519 - 295 = 224 nonzeros. Who are they?

lasso_nonzeros = list(X_train_ks.iloc[:,LASSO_t[0].coef_ != 0])

#We can therefore create and save the PLKS dataset.

X_train_plks_stand = X_train_ks[lasso_nonzeros]

X_test_plks_stand = X_test_ks[lasso_nonzeros]

path = 'C:\\Some\\Local\\Path\\'

X_train_plks_stand.to_csv(path + 'X_train_plks_stand.csv')

X_test_plks_stand.to_csv(path + 'X_test_plks_stand.csv')

