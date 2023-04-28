######################################################################
##SECOND SCRIPT - ONE-HOT-ENCODING, FURTHER CLEANING OF EXTENDED SET##
######################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(1123581321)

scaler = StandardScaler()

'''
COMMENTS

This is the second script regarding the Extended Set 
producing the results in "Human Wellbeing and Machine Learning" by Ekaterina Oparina* (r) Caspar Kaiser* (r) Niccolò Gentile*; 
Alexandre Tkatchenko, Andrew E. Clark, Jan-Emmanuel De Neve 
and Conchita D'Ambrosio (* equal first authors in random order).

Aim of this script is to perform one-hot-encoding of the 
categorival variables, plus some additional cleanings.
'''

path = 'C:\\Users\\niccolo.gentile\\Desktop\\Joined_paper\\A_kitchen_sink_approach_21022022\\'

ks_soep = pd.read_csv(path + 'SOEP_KS_dataset_clean.csv')

ks_soep.drop(['Unnamed: 0'], axis = 1, inplace = True)

#Some additional droppgig of subjective variables.

#As specified in the paper, we exclude "direct measures of 
#subjective wellbeing (such as domain satisfaction, happiness 
#and subjective health) and mental health" 

subj = ['ple0008', 'plh0032','plh0033','plh0034','plh0035','plh0036','plh0037',
        'plh0038','plh0039','plh0040','plh0042', 'plh0162','plh0166','plh0171',
        'plh0172','plh0173','plh0174','plh0175','plh0176','plh0177','plh0178',
        'plh0180','plh0183','plh0184','plh0185','plh0186','plh0187', 'plh0195',
        'plh0188','plh0189','plh0190','plh0191','plh0192','plh0193','plh0194',
        'plh0196','plh0268','plh0272','plh0273_h','plh0274','plj0046', 'plj0047',
        'plj0587','plj0588','plj0589','m11125', 'm11126', 'p11101','ple0026',
        'ple0027','ple0028','ple0029','plh0164','plh0244','hlf0151','plc0111',
        'plc0113','plc0112','plc0114','plh0155', 'ple0033', 'ple0034']

ks_soep.drop(subj, axis = 1, inplace = True)

ks_soep.rename({'plh0182': 'lsat'}, inplace = True)

#To perform the one-hot-encoding procedure, we also consider dummy 
#variables having NaNs as three-classes categorical variables. 

#The ordinal categorical variables are left as they are.

categoricals_1 = ['plh0333','plb0035','pmonin','ple0003','e11106','plh0258_h','plb0022_h',
                  'plh0273_h','pld0224','d11105','d11104','plb0193','plc0011','plb0443',
                  'plb0097','plb0037_h','plb0031_h','hlf0001_h','plj0116_h','plj0043','plj0044',
                  'ple0161','ple0004','hlf0076','ple0164','hlf0011_h','hlf0173_h','plb0103',
                  'migback','ple0165','hlf0241_h','hlf0261','plb0095','hlf0073','ple0005',
                  'plh0039','ple0009','plj0024_h','plj0022','plh0272','plh0268','plh0196',
                  'plh0195','plh0040','plc0440','hlf0087_h','pld0046','plh0038','plh0037',
                  'plh0036','plh0035','plh0034','plh0033','plh0032','plj0046','plj0047','plc0432',
                  'plj0175','plj0105','plb0024_h','iwar2','plb0040','plb0041','plb0156_v1','h11103',
                  'plc0055','e11103','d11108','hlf0092_h','hlf0159','hlf0163_h','hlf0214','hlf0217',
                  'hlf0228','hlc0116','plj0089','plj0173','plc0328','m11113','hlc0121','hlc0124',
                  'hld0002','plc0311','plc0315','plc0342','ple0036','plc0349','plc0363','plc0370','plc0421',
                  'plj0104','ple0081_h','plc0364','hlf0236','hlk0044','plj0151','e11102','e11104',
                  'h11112','l11101_ew','l11102','m11101','m11105','m11106','m11107','m11108','m11109',
                  'm11110','m11111','m11115','hlf0438','m11117','d11102ll','plj0063','m11124','plj0062',
                  'plb0018','plb0020_h','plb0021','plb0282_h','plb0393','plc0015_h','plc0116','plc0446','pld0159',
                  'ple0040','ple0052','ple0053','ple0097','ple0160','plg0012','plh0011_h','plj0009','plj0060',
                  'plj0061','m11119','m11116','hlf0024','hlc0113_h','hlf0037','hlf0036','hlf0035','hlf0034',
                  'hlf0033','hlf0032','hlc0119_h','hlf0029','hlf0006','hlf0018','hlf0031','hlf0030','hlf0025',
                  'hlf0026','hlf0056','hlf0105','hlf0165_h','hlc0098','hlc0052','hlc0007','hlf0178_h','hlf0180',
                  'hlf0182','hlf0184','hlf0186','hlf0188','hlf0190','hlf0192','hlf0194','hlf0239_h','hlf0291','hlf0028',
                  'ispou','y11101','ilib1','chsub','irie1','irie2','kidy','iwith','iaus1','iaus2',
                  'asyl','edupac','ilib2','ichsu','iagr2','ismp2','iagr1','ismp1','sphlp','syear',
                  'icomp','ieret','iunay','iprvp']

categoricals_2 = list(set(categoricals_1) - set(subj))

del categoricals_1

#Renaming for easiness
 
for i in range(len(categoricals_2)):
        
    if categoricals_2[i] == 'pmonin':
        
        categoricals_2[i] = 'month'
        
    if categoricals_2[i] == 'plh0258_h':
        
        categoricals_2[i] = 'religion'
        
    if categoricals_2[i] == 'migback':
        
        categoricals_2[i] = 'ethnicity'
        
    if categoricals_2[i] == 'd11104':
        
        categoricals_2[i] = 'maritalstat'
        
    if categoricals_2[i] == 'hlf0001_h':
        
        categoricals_2[i] = 'homeowner'
        
    if categoricals_2[i] == 'e11102':
        
        categoricals_2[i] = 'empstat'
        
    if categoricals_2[i] == 'l11101_ew':
        
        categoricals_2[i] = 'state'
    
    if categoricals_2[i] == 'd11102ll':
        
        categoricals_2[i] = 'female'
    
    if categoricals_2[i] == 'm11124':
        
        categoricals_2[i] = 'disabled'
    
        
ks_soep.isna().sum().sum()

#584217

ks_soep[categoricals_2].isna().sum().sum()

#584217, implying that all the NaNs are in the categoricals. 

list_of_uniques = []

#Now, we also delete variables that have only one valid value,
#hence useless in the algorithms.

for i in list(ks_soep):
        
    if ks_soep[i].nunique() == 1 or ks_soep[i].isna().sum() == len(ks_soep) or (ks_soep[i].nunique() == 2 and np.sum(ks_soep[i] < 0) > 0):
        
        #In the above:
        
        #ks_soep[i].nunique() == 1 is controlling that there aren't variable with either one valid value
        #only or one value and nans.
        
        #ks_soep[i].isna().sum() == len(ks_soep) is controlling that there are no variables with only nans.
        
        #(ks_soep[i].nunique() == 2 and np.sum(ks_soep[i] < 0) > 0) is controlling that there aren't
        #variables with one value and negative, hence missing.
        
        print('Variable ' + i + ' has only one value, hence useless')
        
        ks_soep.drop(i, axis = 1, inplace = True)
                
        list_of_uniques.append(i)
        
categoricals_3 = [x for x in categoricals_2 if x not in list_of_uniques] 

del categoricals_2

binaries = []        

for i in categoricals_3:
    
    if ks_soep[i].nunique() == 2:
        
        print(i + ' is binary!')
        
        binaries.append(i)
           
categoricals_4 = [x for x in categoricals_3 if x not in binaries]

del categoricals_3       

#For each of the categorical variables, we also extract the
#most populous category (reference category). 

#The most populous category could well be the NaN one (hence the
#dropna = False below).                 

most_pop_cats = []

for i in categoricals_4:
    
    most_pop_cats.append(i + '_' + str(ks_soep[i].mode(dropna = False)[0]))

#Are we sure that in most_pop_cats we 
#have the most populous category for each categorical?
    
len(most_pop_cats) == len(categoricals_4)   

#True. 

########################
##ONE - HOT - ENCODING##
########################

one_hot = pd.get_dummies(ks_soep[categoricals_4], columns = categoricals_4, dummy_na = True)

ks_soep.drop(categoricals_4, axis = 1, inplace = True)

ks_soep_ohed = pd.concat([ks_soep, one_hot], axis = 1)

#In some cases, in extracting the mode, the code adds a .0, whereas
#sometimes it doesn't. Since all the variables in oheing have a .0 in the end
#of the name, we add it if it is not there already and the most populous category
#is not a nan.

most_pop_cats_correct = [x + '.0' for x in most_pop_cats if x[-2:] != '.0' and x[-2:] != 'an']

#then, we create a separate list for the mislabelled variables by the mode function.

most_pop_cats_not_correct = [x for x in most_pop_cats if x[-2:] != '.0' and x[-2:] != 'an']

#and we delete them from the most_pop_cats

most_pop_cats_1 = [x for x in most_pop_cats if x not in most_pop_cats_not_correct]

#to finally instead attach the correct ones.

most_pop_cats_2 = most_pop_cats_1 + most_pop_cats_correct

ks_soep_ohed.drop(most_pop_cats_2, axis = 1, inplace = True)

#The dataset is now ready. The variables that need one-hot-encoding have been ohed,
#and the reference categories dropped.

ks_soep_ohed.isna().sum().sum()

#215330

#Notice that since we aren't oheing the categorical
#ordinal, there are missingness of the kind -2 and nans. 
#We impute them.

#############################
##REMAINING MEAN IMPUTATION##
#############################

#We first check, for security, that indeed 
#all the negative remaining missings are of the kind -2.

for i in list(ks_soep_ohed):
    
    if np.sum(ks_soep_ohed[i] < 0) > 0:
        
        which_neg = [x for x in list(ks_soep_ohed[i].unique()) if x < 0]
        
        print([i, which_neg])

#Indeed yes. Only negative non -2 are -0.0309025, seen 777 times in pli0011,
#being Hours Sundays Running Errands, and -0.34657359027997275 observed 4 times
#in lnhhinc. 

#We impute the 777 in pli0011 (which can't be negative), 
#and leave the 4 negative log incomes.
        
ks_soep_ohed['pli0011'][ks_soep_ohed['pli0011'] < 0] = np.nan

ks_soep_ohed[ks_soep_ohed == -2] = np.nan

for i in list(ks_soep_ohed):
    
    if ks_soep_ohed[i].nunique() == 2:
        
        ks_soep_ohed[i].fillna(ks_soep_ohed[i].mode().iloc[0], inplace = True)
        
    else:
        
        ks_soep_ohed[i].fillna(ks_soep_ohed[i].mean(), inplace = True)
        
ks_soep_ohed.isna().sum().sum()

#0

#######################
###TRAIN - TEST SPLIT##
#######################

#We can proceed with a train-test split, standardization, and
#saving the data.

y = ks_soep_ohed['lsat']

X = ks_soep_ohed.drop(['lsat'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size = 0.20,
                                                    random_state = 1123581321)

X_train_pids = X_train["pid"]

X_test_pids = X_test["pid"]

X_train_to_stand = X_train.drop(["pid"], axis = 1)

X_test_to_stand = X_test.drop(["pid"], axis = 1)

X_train_stand = pd.DataFrame(scaler.fit_transform(X_train_to_stand), index = y_train.index)

X_test_stand = pd.DataFrame(scaler.transform(X_test_to_stand), index =y_test.index)

X_train_stand.columns = list(X_train_to_stand)

X_test_stand.columns = list(X_test_to_stand)

X_train_stand["pid"] = X_train_pids

X_test_stand["pid"] = X_test_pids

dest_path = 'C:\\Some\\Local\\Path\\'    

#As final step, we delete constant columns.

for i in list(X_train_stand):
        
        if X_train_stand[i].nunique() == 1:
            
            X_train_stand.drop(i, axis = 1, inplace = True)
            
            X_test_stand.drop(i, axis = 1, inplace = True)

train = pd.concat([y_train, X_train_stand], axis = 1)
    
test = pd.concat([y_test, X_test_stand], axis = 1)
        
train.to_csv(dest_path + 'train_ks_stand.csv', index = False)
    
test.to_csv(dest_path + 'test_ks_stand.csv', index = False)

