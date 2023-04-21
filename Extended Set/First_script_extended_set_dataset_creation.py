#################################################
##FIRST SCRIPT - DATASET CREATION, EXTENDED SET##
#################################################

'''
COMMENTS

This is the first script regarding the Extended Set 
producing the results in  "Machine Learning in the 
Prediction of Human Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., and et al.

In this script, all the data engineering and wrangling operations
needed to create the SOEP extended set are outlined.

The script is long and encompasses multiple operations: for this
reason, we decided to reduce the amount of comments in it.

This is an adaptation of the script initially on Jupyter Notebook.

We thank Filippo Volpin for excellent assistance in 
helping with this script.
'''

import os
import pandas as pd
import numpy as np
import time
from random import randint
import pylab as pl
from collections import Counter
import math
import statistics
import pyreadstat
import matplotlib.pyplot as plt

#To look up variables: https://paneldata.org/search/all

path = 'C:\\Here\\Some\\Local\\Path\\'

#####################
##IMPORTING FROM pl##
#####################

start = start_total = time.time()

df_pl_234 = pyreadstat.read_file_multiprocessing(pyreadstat.read_dta,
                                                 file_path = f"{path}/datasets/pl_sorted.dta",
                                                 num_processes = 8,
                                                 encoding = 'utf-8',
                                                 row_offset = 452440,
                                                 row_limit = 82146) # 534586 - 452440
end = time.time()

#Keeping the first element as the pyreadstat 
#function returns a tuple, and we don't need the second value

df_pl_234 = df_pl_234[0] 

df_pl_234.drop(df_pl_234[df_pl_234['syear'].isin([2011,2015])].index, inplace = True)

df_pl_234.reset_index(drop = True, inplace = True)

for i,birth in enumerate(df_pl_234['ple0010_h']): 
    
    #Calculate AGE from 'syear' and 'ple0010_h', "year of birth".
    #If 'ple0010_h' is negative, 
    #then assign 999999 (don't worry, 'var_age' 
    #is being dropped at this stage, if var_age < 18)
    
    if birth < 0:     
                                                          
        df_pl_234.loc[i,'var_age'] = 999999
        
    else:
        
        df_pl_234.loc[i,'var_age'] = df_pl_234.loc[i,'syear'] - df_pl_234.loc[i,'ple0010_h']

# drop observations where AGE < 18   
 
df_pl_234.drop(df_pl_234[df_pl_234['var_age'].lt(18)].index, inplace = True)  
  
df_pl_234.drop(columns = ['var_age'], inplace = True)

df_pl_234.reset_index(drop = True, inplace = True)

print(f"\n\ndf_pl_234 (time: {str(round(end - start, 2))} seconds)")


#########################
##IMPORTING FROM pequiv##
#########################

start = time.time()

df_pequiv_234 = pyreadstat.read_file_multiprocessing(pyreadstat.read_dta,
                                                     file_path = f"{path}/datasets/pequiv_sorted.dta",
                                                     num_processes = 8,
                                                     encoding = 'utf-8',
                                                     row_offset = 606208,
                                                     row_limit = 126197) # 732405 - 606208
end = time.time()
    
df_pequiv_234 = df_pequiv_234[0] 

df_pequiv_234.drop(df_pequiv_234[df_pequiv_234['syear'].isin([2011,2015])].index, inplace = True)

#d11101 is "Age of individual", directly available in pequiv 
#Drop observations where age < 18

df_pequiv_234.drop(df_pequiv_234[(df_pequiv_234['d11101'].ge(0)) & (df_pequiv_234['d11101'].lt(18))].index, inplace = True) 

df_pequiv_234.reset_index(drop = True, inplace = True)

print(f"\n\ndf_pequiv_234 (time: {str(round(end - start, 2))} seconds)")

#####################
##IMPORTING FROM hl##
#####################

start = time.time()

df_hl_234 = pyreadstat.read_file_multiprocessing(pyreadstat.read_dta,
                                                 file_path = f"{path}/datasets/hl_sorted.dta",
                                                 num_processes = 8,
                                                 encoding = 'utf-8',
                                                 row_offset = 240012,
                                                 row_limit = 48133) # 288145 - 240012

end = time.time()

df_hl_234 = df_hl_234[0] 

df_hl_234.drop(df_hl_234[df_hl_234['syear'].isin([2011,2015])].index, inplace = True)

df_hl_234.reset_index(drop = True, inplace = True)

print(f"\n\ndf_hl_234 (time: {str(round(end - start, 2))} seconds)")

#########################
##IMPORTING FROM health##
#########################

start = time.time()

df_health_234 = pyreadstat.read_file_multiprocessing(pyreadstat.read_dta,
                                                     file_path = f"{path}/datasets/health_sorted.dta",
                                                     num_processes = 8,
                                                     encoding = 'utf-8',
                                                     row_offset = 454369,
                                                     row_limit = 83766) # 538135 - 454369
end = time.time()
    
df_health_234 = df_health_234[0]

df_health_234.drop(df_health_234[df_health_234['syear'].isin([2011,2015])].index, inplace = True)

df_health_234.reset_index(drop = True, inplace = True)

df_health_234 = df_health_234[['pid','syear','bmi']]

print(f"\n\ndf_health_234 (time: {str(round(end - start, 2))} seconds) # NB: I am only keeping the 'bmi' column!")


########################
##IMPORTING FROM ppath##
########################

start = time.time()

df_ppath = pyreadstat.read_file_multiprocessing(pyreadstat.read_dta,
                                                 file_path = f"{path}/datasets/ppath.dta",
                                                 num_processes = 8,
                                                 encoding = 'utf-8')
end = time.time()

df_ppath = df_ppath[0] 

df_ppath.reset_index(drop = True, inplace = True)

df_ppath = df_ppath[['pid','migback']]

print(f"\n\ndf_ppath (time: {str(round(end - start, 2))} seconds) # NB: I am only keeping the 'migback' column!")

# ---------------------

end_total = time.time()
print(f"Total time: {str(round(end_total - start_total, 2))} seconds")


print(f"{'PL:':>30} {len(set(df_pl_234['pid']))}")
print(f"{'PEQUIV:':>30} {len(set(df_pequiv_234['pid']))}")
print(f"{'PL & PEQUIV:':>30} {len(set(df_pl_234['pid']) & set(df_pequiv_234['pid']))}\n")

print(f"{'PL:':>30} {len(set(df_pl_234['hid']))}")
print(f"{'HL:':>30} {len(set(df_hl_234['hid']))}")
print(f"{'PL & HL:':>30} {len(set(df_pl_234['hid']) & set(df_hl_234['hid']))}\n")

print(f"{'PL:':>30} {len(set(df_pl_234['pid']))}")
print(f"{'HEALTH:':>30} {len(set(df_health_234['pid']))}")
print(f"{'PL & HEALTH:':>30} {len(set(df_pl_234['pid']) & set(df_health_234['pid']))}\n")

print(f"{'PL:':>30} {len(set(df_pl_234['pid']))}")
print(f"{'PPATH:':>30} {len(set(df_ppath['pid']))}")
print(f"{'PL & PPATH:':>30} {len(set(df_pl_234['pid']) & set(df_ppath['pid']))}\n")

#Check which interview date (year) are present in each dataset

print(f"{set(df_pl_234['syear'])}")
print(f"{set(df_pequiv_234['syear'])}")
print(f"{set(df_hl_234['syear'])}")
print(f"{set(df_health_234['syear'])}\n")


df_lookup = pd.read_excel(f"{path}/01_SOEP_KS_lookup.xlsx").drop(columns = ["N_overall","N_dataset"])

#Throughout the all operations, variables that
#are part of the restricted set should never be dropped.

#For this reason, they're also called "protected" down the road.

list_protec_vars = ['d11102ll','d11101','d11106','d11107',
                    'd11109','i11102','m11124','m11127',
                    'd11104','l11101_ew', 'bmi','e11102',
                    'e11101','hlf0001_h','pmonin','plh0258_h',
                    'migback'] 

df_protec_variables = df_lookup[df_lookup["Variable"].isin(list_protec_vars)].reset_index(drop = True)

df_mapping = pd.DataFrame({'Variable': list_protec_vars}).reset_index().set_index('Variable')

df_protec_variables['Var_num'] = df_protec_variables['Variable'].map(df_mapping['index'])

df_protec_variables = df_protec_variables.sort_values(by = 'Var_num').drop(columns = ['Var_num']).reset_index(drop = True)


#Checking the values inside the datasets

print(f"{'df_pl_234:':>18}\t{set(df_pl_234.dtypes)}")
print(f"{'df_pequiv_234:':>18}\t{set(df_pequiv_234.dtypes)}")
print(f"{'df_hl_234:':>18}\t{set(df_hl_234.dtypes)}")
print(f"{'df_health_234:':>18}\t{set(df_health_234.dtypes)}")
print(f"{'df_ppath:':>18}\t{set(df_ppath.dtypes)}")


#Dropping variables containing objects from all the imported
#datasets.

objects = []

for i in list(df_pl_234):
    
    if df_pl_234[i].dtypes == 'O':
        
        objects.append(i)

print(f"{'df_pl_234:':>18}\t{set(df_pl_234.shape)} (before)")

df_pl_234.drop(objects, axis = 1, inplace = True)

print(f"{'df_pl_234:':>18}\t{set(df_pl_234.shape)} (after)\n")

objects = []

for i in list(df_pequiv_234):
    
    if df_pequiv_234[i].dtypes == 'O':
        
        objects.append(i)
            
print(f"{'df_pequiv_234:':>18}\t{set(df_pequiv_234.shape)} (before)")

df_pequiv_234.drop(objects, axis = 1, inplace = True)

print(f"{'df_pequiv_234:':>18}\t{set(df_pequiv_234.shape)} (after)\n")

objects = []

for i in list(df_hl_234):
    
    if df_hl_234[i].dtypes == 'O':
        
        objects.append(i)
            
print(f"{'df_hl_234:':>18}\t{set(df_hl_234.shape)} (before)")

df_hl_234.drop(objects, axis = 1, inplace = True)

print(f"{'df_hl_234:':>18}\t{set(df_hl_234.shape)} (after)")

del objects


#Checking for the presence of 'na' values

print(f"{'df_pl_234:':>18}\t{df_pl_234.isna().sum().sum()}")

print(f"{'df_pequiv_234:':>18}\t{df_pequiv_234.isna().sum().sum()}")

print(f"{'df_hl_234:':>18}\t{df_hl_234.isna().sum().sum()}")

print(f"{'df_health_234:':>18}\t{df_health_234.isna().sum().sum()}")

print(f"{'df_ppath:':>18}\t{df_ppath.isna().sum().sum()}")


#Checking for the presence of empty '' values, across all
#imported datasets.

empty_vars = []

for i in list(df_pl_234):
    
    if np.sum(df_pl_234[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_pl_234:':>18}\t{len(empty_vars)}")

empty_vars = []

for i in list(df_pequiv_234):
    
    if np.sum(df_pequiv_234[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_pequiv_234:':>18}\t{len(empty_vars)}")

empty_vars = []

for i in list(df_hl_234):
    
    if np.sum(df_hl_234[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_hl_234:':>18}\t{len(empty_vars)}")

empty_vars = []

for i in list(df_health_234):
    
    if np.sum(df_health_234[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_health_234:':>18}\t{len(empty_vars)}")

empty_vars = []

for i in list(df_ppath):
    
    if np.sum(df_ppath[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_ppath:':>18}\t{len(empty_vars)}")


#Dropping observations with negative life satisfaction in 
#df_pl_234

print(f"{'df_pl_234:':>18}\t{df_pl_234.shape} (before)")

df_pl_234 = df_pl_234[df_pl_234['plh0182'] > -1]

print(f"{'df_pl_234:':>18}\t{df_pl_234.shape} (after)")

#Separating the necessary years for each of the imported 
#datasets.

df_pl_2 = df_pl_234[df_pl_234['syear'] == 2012]
df_pl_3 = df_pl_234[df_pl_234['syear'] == 2013]
df_pl_4 = df_pl_234[df_pl_234['syear'] == 2014]

df_pequiv_2 = df_pequiv_234[df_pequiv_234['syear'] == 2012]
df_pequiv_3 = df_pequiv_234[df_pequiv_234['syear'] == 2013]
df_pequiv_4 = df_pequiv_234[df_pequiv_234['syear'] == 2014]

df_hl_2 = df_hl_234[df_hl_234['syear'] == 2012]
df_hl_3 = df_hl_234[df_hl_234['syear'] == 2013]
df_hl_4 = df_hl_234[df_hl_234['syear'] == 2014]

df_health_2 = df_health_234[df_health_234['syear'] == 2012]
df_health_3 = df_health_234[df_health_234['syear'] == 2013]
df_health_4 = df_health_234[df_health_234['syear'] == 2014]

#No need to make such subselection in 'df_ppath'
#as it is cross-wave data.

del df_pl_234, df_pequiv_234, df_hl_234, df_health_234

#We create the yearly dataset inner-joining accordingly
#the imported datasets above.

########
##2012##
########

df_2012 = df_pl_2.merge(df_pequiv_2,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_pequiv_delete'))

df_2012 = df_2012.merge(df_hl_2,
                        how = 'inner',
                        left_on = 'hid',
                        right_on = 'hid',
                        suffixes = (None, '_hl_delete'))

df_2012 = df_2012.merge(df_health_2,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_health_delete'))

df_2012 = df_2012.merge(df_ppath,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_ppath_delete'))


list_cols_to_delete = [col for col in df_2012.columns if col[-7:] == "_delete"]

print(f"\n{len(list_cols_to_delete)} variables are in common. They will be deleted, as they contain the same values:\n\n{list_cols_to_delete}\n")

df_2012.drop(list_cols_to_delete, axis = 1, inplace = True)

print(f"{'df_pl_2:':>18}\t{df_pl_2.shape}")
print(f"{'df_pequiv_2:':>18}\t{df_pequiv_2.shape}")
print(f"{'df_hl_2:':>18}\t{df_hl_2.shape}")
print(f"{'df_health_2:':>18}\t{df_health_2.shape}")
print(f"{'df_ppath:':>18}\t{df_ppath.shape}\n")

print(f"{'df_2012:':>18}\t{df_2012.shape}\n")

A = len(list(df_pl_2))
B = len(list(df_pequiv_2))
C = len(list(df_hl_2))
D = len(list(df_health_2))
X = len(list(df_ppath))

E = len(list_cols_to_delete)

print(f"DOUBLE CHECK:")
print(f"{A} + {B} + {C} + {D} + {X} - {1} - {1} - {1} - {1} - {E} = {A + B + C + D + X - 1 - 1 - 1 - 1 - E} variables\n") 

# the (-1)s are because the [hipd/pidp] variable is 
#shared by the datasets being merged

#Checking for the presence of 'na' values

print(f"{'df_2012:':>18}\t{df_2012.isna().sum().sum()}\n")

#Checking for the presence of empty '' values

empty_vars = []

for i in list(df_2012):
    
    if np.sum(df_2012[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_2012:':>18}\t{len(empty_vars)}")

########
##2013##
########

df_2013 = df_pl_3.merge(df_pequiv_3,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_pequiv_delete'))

df_2013 = df_2013.merge(df_hl_3,
                        how = 'inner',
                        left_on = 'hid',
                        right_on = 'hid',
                        suffixes = (None, '_hl_delete'))

df_2013 = df_2013.merge(df_health_3,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_health_delete'))

df_2013 = df_2013.merge(df_ppath,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_ppath_delete'))


list_cols_to_delete = [col for col in df_2013.columns if col[-7:] == "_delete"]

print(f"\n{len(list_cols_to_delete)} variables are in common. They will be deleted, as they contain the same values:\n\n{list_cols_to_delete}\n")

df_2013.drop(list_cols_to_delete, axis = 1, inplace = True)


print(f"{'df_pl_3:':>18}\t{df_pl_3.shape}")
print(f"{'df_pequiv_3:':>18}\t{df_pequiv_3.shape}")
print(f"{'df_hl_3:':>18}\t{df_hl_3.shape}")
print(f"{'df_health_3:':>18}\t{df_health_3.shape}")
print(f"{'df_ppath:':>18}\t{df_ppath.shape}\n")

print(f"{'df_2013:':>18}\t{df_2013.shape}\n")

A = len(list(df_pl_3))
B = len(list(df_pequiv_3))
C = len(list(df_hl_3))
D = len(list(df_health_3))
X = len(list(df_ppath))

E = len(list_cols_to_delete)

print(f"DOUBLE CHECK:")
print(f"{A} + {B} + {C} + {D} + {X} - {1} - {1} - {1} - {1} - {E} = {A + B + C + D + X - 1 - 1 - 1 - 1 - E} variables\n") # the (-1)s are because the [hipd/pidp] variable is shared by the two datasets being merged

#Checking for the presence of 'NA' values

print(f"{'df_2013:':>18}\t{df_2013.isna().sum().sum()}\n")


#Checking for the presence of empty '' values

empty_vars = []

for i in list(df_2013):
    
    if np.sum(df_2013[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_2013:':>18}\t{len(empty_vars)}")

########
##2014##
########

df_2014 = df_pl_4.merge(df_pequiv_4,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_pequiv_delete'))

df_2014 = df_2014.merge(df_hl_4,
                        how = 'inner',
                        left_on = 'hid',
                        right_on = 'hid',
                        suffixes = (None, '_hl_delete'))

df_2014 = df_2014.merge(df_health_4,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_health_delete'))

df_2014 = df_2014.merge(df_ppath,
                        how = 'inner',
                        left_on = 'pid',
                        right_on = 'pid',
                        suffixes = (None, '_ppath_delete'))

list_cols_to_delete = [col for col in df_2014.columns if col[-7:] == "_delete"]

print(f"\n{len(list_cols_to_delete)} variables are in common. They will be deleted, as they contain the same values:\n\n{list_cols_to_delete}\n")

df_2014.drop(list_cols_to_delete, axis = 1, inplace = True)


print(f"{'df_pl_4:':>18}\t{df_pl_4.shape}")
print(f"{'df_pequiv_4:':>18}\t{df_pequiv_4.shape}")
print(f"{'df_hl_4:':>18}\t{df_hl_4.shape}")
print(f"{'df_health_4:':>18}\t{df_health_4.shape}")
print(f"{'df_ppath:':>18}\t{df_ppath.shape}\n")


print(f"{'df_2014:':>18}\t{df_2014.shape}\n")

A = len(list(df_pl_4))
B = len(list(df_pequiv_4))
C = len(list(df_hl_4))
D = len(list(df_health_4))
X = len(list(df_ppath))

E = len(list_cols_to_delete)

print(f"DOUBLE CHECK:")
print(f"{A} + {B} + {C} + {D} + {X} - {1} - {1} - {1} - {1} - {E} = {A + B + C + D + X - 1 - 1 - 1 - 1 - E} variables\n") # the (-1)s are because the [hipd/pidp] variable is shared by the two datasets being merged



#Checking for the presence of 'na' values

print(f"{'df_2014:':>18}\t{df_2014.isna().sum().sum()}\n")


#Checking for the presence of empty '' values

empty_vars = []

for i in list(df_2014):
    
    if np.sum(df_2014[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_2014:':>18}\t{len(empty_vars)}")


#Checking the values inside the datasets

print(f"{'df_2012:':>18}\t{set(df_2012.dtypes)}")
print(f"{'df_2013:':>18}\t{set(df_2013.dtypes)}")
print(f"{'df_2014:':>18}\t{set(df_2014.dtypes)}")


#Printing the shape of the datasets: (observations x variables)

print(f"{'df_2012:':>18}\t{df_2012.shape[0]:6} x{df_2012.shape[1]:6}")
print(f"{'df_2013:':>18}\t{df_2013.shape[0]:6} x{df_2013.shape[1]:6}")
print(f"{'df_2014:':>18}\t{df_2014.shape[0]:6} x{df_2014.shape[1]:6}")

#Are there any unavailable variables?
print(f'{set(list_protec_vars) - set(df_2012[df_2012.columns[df_2012.columns.isin(list_protec_vars)]].columns)}')


#Are there any unavailable variables?
print(f'{set(list_protec_vars) - set(df_2013[df_2013.columns[df_2013.columns.isin(list_protec_vars)]].columns)}')


#Are there any unavailable variables?
print(f'{set(list_protec_vars) - set(df_2014[df_2014.columns[df_2014.columns.isin(list_protec_vars)]].columns)}')

#Check which interview date (year) are present in each dataset

print(f"{set(df_2012['syear'])}")
print(f"{set(df_2013['syear'])}")
print(f"{set(df_2014['syear'])}\n")

df_unpredictive = pd.read_excel(f"{path}/02_SOEP_KS_variables_predictive_content.xlsx").drop(columns = ['N_overall','N_dataset'])

#KEY

#NaN = drop
#1 = keep
#2 = Do a check

#Drop everything that is not 'keep'

list_drop_variables = df_unpredictive[df_unpredictive['KEY'].ne(1)]['VARIABLE'].values 

#Make sure to remove protected variables from dropping list!

list_drop_variables = list(set(list_drop_variables) - set(list_protec_vars))

for i,var in enumerate(list_drop_variables):
    
    print(f"{i+1:5}.\t{var:30}{df_lookup[df_lookup['Variable'] == var]['Label'].values[0]}")

print(f"df_2012:\t{df_2012.shape} (before)")

df_2012.drop(columns = list_drop_variables, errors = "ignore", inplace = True)

print(f"df_2012:\t{df_2012.shape} (after)\n")

print(f"df_2013:\t{df_2013.shape} (before)")

df_2013.drop(columns = list_drop_variables, errors = "ignore", inplace = True)

print(f"df_2013:\t{df_2013.shape} (after)\n")

print(f"df_2014:\t{df_2014.shape} (before)")

df_2014.drop(columns = list_drop_variables, errors = "ignore", inplace = True)

print(f"df_2014:\t{df_2014.shape} (after)\n")

#Resetting the columns indexes of each dataset

df_2012.reset_index(drop = True, inplace = True)
df_2013.reset_index(drop = True, inplace = True)
df_2014.reset_index(drop = True, inplace = True)

#Here, we measure the impact of changing the 
#threshold for dropping columns based on the degree of 
#nmissingness between 0.05 and 0.5.

list_thresholds = range(5,100,5)
list_x = []
list_y = []

for thres in list_thresholds:
    thres = thres / 100
    count_drops = 0
    for i in list(df_2013):
        if np.sum(df_2013[i] < 0) >= thres * len(df_2013[i]):
            count_drops += 1
    list_x.append(thres)
    list_y.append(count_drops)
    print(f"Threshold: {thres:2}\t\tVars dropped: {count_drops:5}\tVars remaining: {len(list(df_2013))-count_drops:5}")


plt.style.use('classic')
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5))
fig.suptitle("(ALL TYPES OF) MISSINGNESS DROP: variables dropped (left) and remaining (right)", fontsize = 16, y = 1.05)

ax1.set_xlabel("Threshold (observations missing more than %)", fontsize = 14)
ax1.set_ylabel("Variables dropped", fontsize = 14)
ax1.set_xticks(ticks = np.arange(0, 1, step = 0.1))
ax1.set_yticks(ticks = np.arange(0, 6000, step = 100))

ax1.plot(list_x, list_y, 'r', alpha = 0.5)
ax1.plot(list_x, list_y, 'rs')
ax1.grid(True, alpha = 0.7)


list_variables_remaining = list(np.subtract(df_2013.shape[1],np.array(list_y)))

ax2.set_xlabel("Threshold (observations missing more than %)", fontsize = 14)
ax2.set_ylabel("Variables remaining", fontsize = 14)
ax2.set_xticks(ticks = np.arange(0, 1, step = 0.1))
ax2.set_yticks(ticks = np.arange(0, 6000, step = 100))

ax2.plot(list_x, list_variables_remaining, 'b', alpha = 0.5)
ax2.plot(list_x, list_variables_remaining, 'bs')
ax2.grid(True, alpha = 0.7)
plt.style.use('default')


#Noice that there are no `NA` or empty `''` values in 
#the dataframes. This means that all "invalid" values 
#are coded with negative entries.
#Let's now check for missing values via counts 
#of negative values (instead of `NA` or empty `''`)

threshold = 0.5

#For each datasets, there are (n) variables that can 
#be dropped because negative (including '-2') in 
#{threshold * 100}% or more observations

list_miss_2012 = []

for i in list(df_2012):
    
    if np.sum(df_2012[i] < 0) >= threshold * len(df_2012[i]):
        
        list_miss_2012.append(i)
        
print(f"list_miss_w2_indhh_joined: ({len(list_miss_2012)})\tthat is, {round(len(list_miss_2012)/(df_2012.shape[1])*100,1)}% of the variables")

list_miss_2013 = []

for i in list(df_2013):
    
    if np.sum(df_2013[i] < 0) >= threshold * len(df_2013[i]):
        
        list_miss_2013.append(i)
        
print(f"list_miss_w2_indhh_joined: ({len(list_miss_2013)})\tthat is, {round(len(list_miss_2013)/(df_2013.shape[1])*100,1)}% of the variables")

list_miss_2014 = []

for i in list(df_2014):
    
    if np.sum(df_2014[i] < 0) >= threshold * len(df_2014[i]):
        
        list_miss_2014.append(i)
        
print(f"list_miss_w2_indhh_joined: ({len(list_miss_2014)})\tthat is, {round(len(list_miss_2014)/(df_2014.shape[1])*100,1)}% of the variables\n\n")

#Make sure to remove protected variables from dropping lists!

##########
###2012###
##########

list_miss_2012 = sorted(list(set(list_miss_2012) - set(list_protec_vars))) 

print(f"\n\nlist_miss_2012 ({len(list_miss_2012)}):\n")

for i,var in enumerate(list_miss_2012):
    
    try:
        
        print(f"{i+1:5}.\t{var:30}{df_lookup[df_lookup['Variable'] == var]['Label'].values[0]}")
        
    except:
        
        print(f"{i+1:5}.\t{var:30}{'ERROR: Variable is not available anymore'}")

##########
###2013###
##########

list_miss_2013 = sorted(list(set(list_miss_2013) - set(list_protec_vars))) 

print(f"\n\nlist_miss_2013 ({len(list_miss_2013)}):\n")

for i,var in enumerate(list_miss_2013):
    
    try:
        
        print(f"{i+1:5}.\t{var:30}{df_lookup[df_lookup['Variable'] == var]['Label'].values[0]}")
        
    except:
        
        print(f"{i+1:5}.\t{var:30}{'ERROR: Variable is not available anymore'}")
        
##########
###2014###
##########

list_miss_2014 = sorted(list(set(list_miss_2014) - set(list_protec_vars))) 

print(f"\n\nlist_miss_2014 ({len(list_miss_2014)}):\n")

for i,var in enumerate(list_miss_2014):
    
    try:
        
        print(f"{i+1:5}.\t{var:30}{df_lookup[df_lookup['Variable'] == var]['Label'].values[0]}")
        
    except:
        
        print(f"{i+1:5}.\t{var:30}{'ERROR: Variable is not available anymore'}")

#Dropping the respective missing variables in each of the three datasets

#How many variables have been dropped?

print(f"{'df_2012:':>18}\t{df_2012.shape[0]:6} x{df_2012.shape[1]:6} (before)")
print(f"{'df_2013:':>18}\t{df_2013.shape[0]:6} x{df_2013.shape[1]:6} (before)")
print(f"{'df_2014:':>18}\t{df_2014.shape[0]:6} x{df_2014.shape[1]:6} (before)\n")

col_count = np.subtract(df_2012.shape, df_2012.drop(list_miss_2012, axis = 1).shape)[1]
print(f"{'df_2012:':>18}\t({col_count}) variables dropped")
df_2012.drop(list_miss_2012, axis = 1, inplace = True)

col_count = np.subtract(df_2013.shape, df_2013.drop(list_miss_2013, axis = 1).shape)[1]
print(f"{'df_2013:':>18}\t({col_count}) variables dropped")
df_2013.drop(list_miss_2013, axis = 1, inplace = True)

col_count = np.subtract(df_2014.shape, df_2014.drop(list_miss_2014, axis = 1).shape)[1]
print(f"{'df_2014:':>18}\t({col_count}) variables dropped\n")
df_2014.drop(list_miss_2014, axis = 1, inplace = True)

del col_count

print(f"{'df_2012:':>18}\t{df_2012.shape[0]:6} x{df_2012.shape[1]:6} (after)")
print(f"{'df_2013:':>18}\t{df_2013.shape[0]:6} x{df_2013.shape[1]:6} (after)")
print(f"{'df_2014:':>18}\t{df_2014.shape[0]:6} x{df_2014.shape[1]:6} (after)")


#Resetting the columns indexes of each dataset

df_2012.reset_index(drop = True, inplace = True)
df_2013.reset_index(drop = True, inplace = True)
df_2014.reset_index(drop = True, inplace = True)

#Checking for the presence of empty '' values

empty_vars = []

for i in list(df_2012):
    
    if np.sum(df_2012[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_2012:':>18}\t{len(empty_vars)}")

empty_vars = []

for i in list(df_2013):
    
    if np.sum(df_2013[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_2013:':>18}\t{len(empty_vars)}")

empty_vars = []

for i in list(df_2014):
    
    if np.sum(df_2014[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_2014:':>18}\t{len(empty_vars)}")

#Checking for the presence of 'na' values

print(f"{'df_2012:':>18}\t{df_2012.isna().sum().sum()}")
print(f"{'df_2013:':>18}\t{df_2013.isna().sum().sum()}")
print(f"{'df_2014:':>18}\t{df_2014.isna().sum().sum()}")


#Checking how many values in our datasets are negative

ratio_negatives = df_2012.lt(0).sum().sum() / (len(df_2012) * len(list(df_2012))) * 100
print(f"{'df_2012:':>18}\t{ratio_negatives:.3} % (ratio_negatives)")

ratio_negatives = df_2013.lt(0).sum().sum() / (len(df_2013) * len(list(df_2013))) * 100
print(f"{'df_2013:':>18}\t{ratio_negatives:.3} % (ratio_negatives)")

ratio_negatives = df_2014.lt(0).sum().sum() / (len(df_2014) * len(list(df_2014))) * 100
print(f"{'df_2014:':>18}\t{ratio_negatives:.3} % (ratio_negatives)")

del ratio_negatives

#Transforming all negative values (except for '-2') to np.nan

df_2012[(df_2012 < 0) & (df_2012 != -2)] = np.nan
df_2013[(df_2013 < 0) & (df_2013 != -2)] = np.nan
df_2014[(df_2014 < 0) & (df_2014 != -2)] = np.nan

#Share of np.nan values

ratio_nans = df_2012.isna().sum().sum() / (len(df_2012) * len(list(df_2012))) * 100
print(f"{'df_2012:':>18}\t{ratio_nans:.3} % (ratio_nans)")

ratio_nans = df_2013.isna().sum().sum() / (len(df_2013) * len(list(df_2013))) * 100
print(f"{'df_2013:':>18}\t{ratio_nans:.3} % (ratio_nans)")

ratio_nans = df_2014.isna().sum().sum() / (len(df_2014) * len(list(df_2014))) * 100
print(f"{'df_2014:':>18}\t{ratio_nans:.3} % (ratio_nans)\n\n")

#Share of '-2' values

ratio_eights = df_2012.eq(-2).sum().sum() / (len(df_2012) * len(list(df_2012))) * 100
print(f"{'df_2012:':>18}\t{ratio_eights:.3} % (ratio_twos)")

ratio_eights = df_2013.eq(-2).sum().sum() / (len(df_2013) * len(list(df_2013))) * 100
print(f"{'df_2013:':>18}\t{ratio_eights:.3} % (ratio_twos)")

ratio_eights = df_2014.eq(-2).sum().sum() / (len(df_2014) * len(list(df_2014))) * 100
print(f"{'df_2014:':>18}\t{ratio_eights:.3} % (ratio_twos)")

del ratio_nans, ratio_eights

print(df_2012[(df_2012 < 0) & (df_2012 != -2)].lt(0).sum().sum())
print(df_2013[(df_2013 < 0) & (df_2013 != -2)].lt(0).sum().sum())
print(df_2014[(df_2014 < 0) & (df_2014 != -2)].lt(0).sum().sum())

#In df_2012 and df_2014, we only keep entries with 'pid' from year 2013
#Reason is that we need values from 2012 and 2014 only to forward
#and backward impute values of 2013.

#How many observation have been dropped?

print(f"{'df_2012:':>18}\t{df_2012.shape[0]:5} x{df_2012.shape[1]:5} (before)")
df_2012 = df_2012[df_2012['pid'].isin(df_2013['pid'])]
print(f"{'df_2012:':>18}\t{df_2012.shape[0]:5} x{df_2012.shape[1]:5} (after)\n")

print(f"{'df_2014:':>18}\t{df_2014.shape[0]:5} x{df_2014.shape[1]:5} (before)")
df_2014 = df_2014[df_2014['pid'].isin(df_2013['pid'])]
print(f"{'df_2014:':>18}\t{df_2014.shape[0]:5} x{df_2014.shape[1]:5} (after)")

#Resetting the columns indexes of each dataset

df_2012.reset_index(drop = True, inplace = True)
df_2013.reset_index(drop = True, inplace = True)
df_2014.reset_index(drop = True, inplace = True)

############################
####LEFT JOIN OPERATIONS####
############################

#df_2013 <-- df_2014, resulting in df_34_temp

df_34_temp = df_2013.merge(df_2014,
                           how = 'left',
                           left_on = ['pid'],
                           right_on = ['pid'],
                           suffixes = (None, "_year2014"))

print(f"{'df_2013:':>18}\t{df_2013.shape} (before)")
print(f"{'df_2014:':>18}\t{df_2014.shape}")
print(f"{'df_34_temp:':>18}\t{df_34_temp.shape} (after)\n\n")


#df_34_temp <-- df_2012, resulting in df_234

df_234 = df_34_temp.merge(df_2012,
                          how = 'left',
                          left_on = ['pid'],
                          right_on = ['pid'],
                          suffixes = (None, "_year2012"))

print(f"{'df_34_temp:':>18}\t{df_34_temp.shape} (before)")
print(f"{'df_2012:':>18}\t{df_2012.shape}")
print(f"{'df_234:':>18}\t{df_234.shape} (after)")

del df_34_temp

#Double check:

#Counting the missing values in the 2013 dataset and the 
#missing values in the three years-joined dataset.

#Selecting only the variables (columns) of 2013, leads to 
#the same result.

print(f"{df_2013.isna().sum().sum()} 'NA' values in df_2013")

imput_NAs = df_234[list(df_2013)].isna().sum().sum()

print(f"{imput_NAs} 'NA' values in df_234[only for df_2013 columns]")

###############################################
###BACKWARD IMPUTATION - PROTECTED VARIABLES###
###############################################

success_count = 0
failure_count = 0
failed_variables = []

for var in list_protec_vars:
    
    var_2014 = "".join([var,'_year2014'])
    
    try:
        
        df_234[var].fillna(df_234[var_2014], inplace = True)
        
        success_count += 1
        
    except:
        
        failure_count += 1
        
        failed_variables.append(var)

print(f"In total, of the {len(list_protec_vars)} protected variables:")
print(f"{success_count} variables have been used for backward-imputation successfully, whereas {failure_count} variables have failed")
print(f"{failed_variables}\n")

imput_back_imput_2014 = imput_NAs - df_234[list(df_2013)].isna().sum().sum()

print(f"Using the observations of the protected variables from w4, an additional {imput_NAs} - {df_234[list(df_2013)].isna().sum().sum()} = {imput_back_imput_2014} values were backward-imputed.\n\n")



###############################################
###FORWARD  IMPUTATION - PROTECTED VARIABLES###
###############################################

success_count = 0
failure_count = 0
failed_variables = []

for var in list_protec_vars:
    
    var_2012 = "".join([var,'_year2012'])
    
    try:
        
        df_234[var].fillna(df_234[var_2012], inplace = True)
        
        success_count += 1
        
    except:
        
        failure_count += 1
        
        failed_variables.append(var)

print(f"In total, of the {len(list_protec_vars)} protected variables:")
print(f"{success_count} variables have been used for forward-imputation successfully, whereas {failure_count} variables have failed")
print(f"{failed_variables}\n")

imput_forw_imput_2012 = imput_NAs - imput_back_imput_2014 - df_234[list(df_2013)].isna().sum().sum()

print(f"Using the observations of the protected variables from w2, an additional {imput_NAs - imput_back_imput_2014} - {df_234[list(df_2013)].isna().sum().sum()} = {imput_forw_imput_2012} values were forward-imputed.")

############################
###IMPUTATION OF RELIGION###
############################

#Importing "pl"


start = time.time()
df_religion = pyreadstat.read_file_multiprocessing(pyreadstat.read_dta,
                                                   file_path = f"{path}/datasets/pl_sorted.dta",
                                                   num_processes = 8,
                                                   encoding = 'utf-8',
                                                   usecols = ['pid','plh0258_h','syear'],
                                                #  row_offset = 425113)
                                                   row_limit = 534585)
end = time.time()

df_religion = df_religion[0] 

df_religion.reset_index(drop = True, inplace = True)

print(f"df_religion (time: {str(round(end - start, 2))} seconds)")

#To understand the imputation strategy, good to keep track
#of the possible values here:
    
#https://paneldata.org/soep-core/data/pl/plh0258_h

#plh0258_h = "Kirche, Religion [harmonisiert]"

# VALUE     ACTION  LABEL

#  1        keep    Catholic
#  2        keep    Evangelical
#  3        keep    Other Christian
#  4        keep    Islamic 
#  5        keep    Other religious community
#  6        keep    Non-denominational
#  7        keep    Christian Orthodox
#  8        keep    Shiite
#  9        keep    Sunni
# 10        keep    Alevi
# 11       <drop>   multiple answers
# -1       <drop>   no information
# -2        keep    does not apply (NA) -- I guess this is (I'm atheist == no religion)
# -5       <drop>   not included in the questionnaire version
# -7       <drop>   only available in less restricted edition
# -8       <drop>   question not part of the questionnaire this year

def fn_custom_mode(my_list):
    
    my_positive_list = [x for x in my_list if x in [1,2,3,4,5,6,7,8,9,10,-2]]
    
    my_mode = statistics.multimode(my_positive_list)
    
    #Conditional statements on my_mode
    
    #If there is more than one mode:

    if len(my_mode) > 1:    

        #Take the highest value
        
        my_mode = max(my_mode) 
        
    #If all values are invalid (have been dropped):  
        
    if my_mode == []:     

       #Assign the value np.nan          
             
        my_mode = np.nan    

    #If my_mode happens to be a list instead of int:
    
    if type(my_mode) == list:       
        
        #fix its type accordingly
        
        my_mode = my_mode[0]
    
    return my_mode

#For each individual, we check which has been her/his religion
#in the past or, if he/she has changed religion, which one
#she/he has professed the most (atheism included).

#This is the same strategy used to impute Religion when creating the
#Restrcted Set.

df_aggregate = df_religion.copy()[['pid','plh0258_h']].groupby('pid').aggregate(fn_custom_mode)

df_aggregate.reset_index(inplace = True)

df_aggregate.rename(columns = {'plh0258_h':'plh0258_h_fedforward'}, inplace = True)

df_234 = df_234.merge(df_aggregate,
                      how = 'left',
                      left_on = 'pid',
                      right_on = 'pid',
                      suffixes = (None, '_delete'))

#And we can therefore impute in the main df_234 the religion
#to anyone who has provided info in the past about their
#belief.

df_234['plh0258_h'].fillna(df_234['plh0258_h_fedforward'], inplace = True)

#For those instead who never provided info about their religion, we 
#simply impute the resulting mode after the previous.

df_234['plh0258_h'].fillna(df_234['plh0258_h'].mode()[0], inplace = True)

df_234.drop(columns = ['plh0258_h_fedforward'], inplace = True)

del df_aggregate
del df_religion

######################################
##DISCARDING 2012 AND 2014 VARIABLES##
######################################
 
list_vars_to_delete = [col for col in df_234.columns if col[-9:] == "_year2012" or col[-9:] == "_year2014"]

print(f"{len(list_vars_to_delete)} columns will be deleted (because they have either '_year2012' or '_year2014' as suffix\n")

print(f"{df_234.shape} (before)")

df_234.drop(list_vars_to_delete, axis = 1, inplace = True)

print(f"{df_234.shape} (after)")

df_soep = df_234.copy()

df_soep.reset_index(drop = True, inplace = True)

#Checking for the presence of empty '' values

empty_vars = []

for i in list(df_soep):
    
    if np.sum(df_soep[i] == '') >= 1:
        
        empty_vars.append(i)
        
print(f"{'df_soep:':>18}\t{len(empty_vars)}\n")

#Checking for the presence of 'na' values

print(f"{'df_soep:':>18}\t{df_soep.isna().sum().sum()}")

ratio_twos = df_soep.eq(-2).sum().sum() / (len(df_soep) * len(list(df_soep))) * 100

print(f"df_soep: {ratio_twos} % (ratio_twos)\n")

ratio_negatives = df_soep.lt(0).sum().sum() / (len(df_soep) * len(list(df_soep))) * 100

print(f"df_soep: {ratio_negatives} % (ratio_negatives)\n")

print(df_soep[(df_soep < 0) & (df_soep != -2)].lt(0).sum().sum())

#Are there any unavailable variables?

print(f'{set(list_protec_vars) - set(df_soep[df_soep.columns[df_soep.columns.isin(list_protec_vars)]].columns)}')

#List of all the variables in 'df_soep'

list_var = []
list_lab = []

for i,var in enumerate(list(df_soep)):
    
    list_var.append(var)
    
    try:
        
        list_lab.append(df_lookup[df_lookup['Variable'] == var]['Label'].values[0])
        
    except:
        
        list_lab.append('ERROR: Variable is not available anymore')

df_soep_variables = (pd.DataFrame([list_var, list_lab]).T).rename(columns = {0:"Variable",1:"Label"})

#Of the protected variables, how many values are missing 
#(i.e., nan) or 'na' (i.e., value '-2')?

(pd.DataFrame(np.array([list_protec_vars,
                        df_protec_variables['Label'].values,
                        df_soep[list_protec_vars].isna().sum(),
                        round(df_soep[list_protec_vars].isna().sum() / len(df_soep) * 100, 2),
                        df_soep[list_protec_vars].eq(-2).sum(),
                        round(df_soep[list_protec_vars].eq(-2).sum() / len(df_soep) * 100, 2)])).T).rename(columns = {0:'protected_variable',
                                                                                                                      1:'label',
                                                                                                                      2:'NaN_count',
                                                                                                                      3:'NaN_share%',
                                                                                                                      4:"NA_'-2'_count",
                                                                                                                      5:"NA_'-2'_share%"})

#To determine which variables are categoricals 
#(and which aren't) idea is to create a table with the number of 
#unique values for each variable

print(f"{'N':>4}. {'Variable':>20} {'Uniques':>8}\t{'Label':80}{'Unique_values'}\n")

list_unique_values_counts = []
list_unique_values = []
list_num_type = []

threshold_more_than_count = 1000

for i,var in enumerate(df_soep_variables['Variable']):

    set_uniques = set([x for x in df_soep[var] if x == x])
    
    #Cast all values to float with three decimal points, 
    #so that infinitely small decimals are not in the way
    
    set_uniques = round(pd.Series(np.array(list(set_uniques))).astype('float'), 3)
    int_flag = 'float'

    #If all decimals are exactly zero, then 
    #convert float --> integer
    
    series_decimals = pd.Series([math.modf(x)[0] for x in set_uniques])
    
    if (series_decimals == 0).sum() == len(series_decimals):
        
        set_uniques = pd.Series(np.array(list(set_uniques))).astype('int')
        
        int_flag = 'int'

    set_uniques = sorted(list(set(set_uniques)))
    
    list_unique_values_counts.append(len(set_uniques))
    
    list_num_type.append(int_flag)

    if len(set_uniques) > threshold_more_than_count:
        
        print(f"{i:>4}. {var:>20} {len(set_uniques):>8}\t{df_soep_variables.loc[i,'Label']:80}{int_flag:>8}\tMore than {threshold_more_than_count} unique values")
        
        list_unique_values.append(f'More than {threshold_more_than_count} unique values')
        
    else:
        
        print(f"{i:>4}. {var:>20} {len(set_uniques):>8}\t{df_soep_variables.loc[i,'Label']:80}{int_flag:>8}\t{set_uniques}")
        
        list_unique_values.append(sorted(list(set_uniques)))

df_unique_value_counts = pd.DataFrame(np.array([df_soep_variables['Variable'],
                                                list_unique_values_counts,
                                                df_soep_variables['Label'],
                                                list_num_type,
                                                list_unique_values])).T.rename(columns = {0:"Variable",
                                                                                          1:"Unique_values_count",
                                                                                          2:"Label",
                                                                                          3:"Num_type",
                                                                                          4:"Unique_values"}).sort_values(by = ['Unique_values_count'], ascending = False)

display(df_unique_value_counts)

save_here = f"{path}/03_SOEP_KS_variable_types.xlsx"

df_unique_value_counts.to_excel(save_here)

print(f" >> Export completed! File exported to '{save_here}'")

counter_uniques = Counter(list_unique_values_counts)

dict_uniques = {k: b for k, b in sorted(dict(counter_uniques).items(), key=lambda element: element[0])}

print(dict_uniques)

list_x = dict_uniques.keys() # Unique count
list_y = dict_uniques.values() # Frequency of unique count

plt.style.use('classic')

fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (45,5))
fig.suptitle(f"Distribution of unique value counts\n\nx : log(y)", fontsize = 20, y = 1.15)


x_pos = [i for i,_ in enumerate(list_x)]

ax1.grid(zorder = 1, alpha = 0.7)
ax1.bar(x_pos,list_y, zorder = 2, alpha = 0.8)

plt.xticks(x_pos, list_x, rotation='vertical')

ax1.set_xlabel("Unique value count", fontsize = 14)
ax1.set_ylabel("How many variables have\nthis unique value count?", fontsize = 14, y = 0.5, rotation = 0)
ax1.xaxis.set_label_coords(0.5, -0.2)
ax1.yaxis.set_label_coords(-0.06, 0.45)
ax1.set_xticks(ticks = np.arange(0, len(list_x), step = 1))
ax1.set_yticks(ticks = np.arange(0, 200, step = 20))
ax1.set_xlim(-2,len(list_x)+2)
ax1.set_ylim(0.6,max(list_y)*1.5)

ax1.set_yscale('log')

plt.show()
plt.style.use('default')

plt.style.use('classic')

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))
fig.suptitle(f"Distribution of unique value counts", fontsize = 20, y = 1.15)

# PLOT 1
list_x = dict_uniques.keys() # Unique count
list_y = dict_uniques.values() # Frequency of unique count

ax1.set_title("log(x) - log(y)", pad = 12)
ax1.set_xlabel("Unique value count")
ax1.set_ylabel("How many variables have\nthis unique value count?", rotation = 0)
ax1.yaxis.set_label_coords(-0.4, 0.45)
ax1.grid(alpha = 0.7, zorder = 1)
ax1.hist(list_y, bins = np.logspace(np.log10(min(list_x)),np.log10(max(list_x)), len(list_y)*2), zorder = 2, alpha = 0.8)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.9,max(list_y)*2)
ax1.set_ylim(min(list_y)/2,max(list_y)*2)

# PLOT 2
del list_x
list_y = dict_uniques.values() # Frequency of unique count

ax2.set_title("x - log(y)", pad = 12)
ax2.set_xlabel("Unique value count")
ax2.grid(alpha = 0.7, zorder = 1)
ax2.hist(list_y, bins = len(list_y), zorder = 2, alpha = 0.8)

ax2.set_xticks(ticks = np.arange(0, 200, step = 10))
ax2.set_xscale('linear')
ax2.set_yscale('log')
ax2.set_xlim(-min(list_y),max(list_y)*1.05)
ax2.set_ylim(0.5,max(list_y)*2)

plt.show()
plt.style.use('default')

df_final = df_soep.copy()

df_types = pd.read_excel(f"{path}/04_SOEP_KS_variable_types_filled.xlsx").drop(columns=['N','Duplicate'])

#######################################
###IMPUTATION OF REMAINING VARIABLES###
#######################################

#             KEY                           IMPUTATION METHOD
dict_types = {1: 'continuous',              # mean
              2: 'discrete continuous',     # mean rounded
              3: 'categorical',
              4: 'ordered categorical',     # mode  (if there is more than one mode, take the larger one)
              -1: 'drop'}
#              -2: 'duplicate'}              # create a single variable from multiple duplicates


#How many variables are in the dataset for each type?

print(f"{'KEY':>8}  {'TYPE':20}   {'N_variables':>7}\n")

for key,type in dict_types.items():
    
    print(f"{key:>8}  {type:20}   {(df_types['Key'] == key).sum():>7}")

#How many 'np.nan' values for each variable type?

print(f"{'KEY':>8}  {'TYPE':20}   {'N_np.nan':>7}\n")

for key,type in dict_types.items():
    
    list_vars = list(df_types[df_types['Insert_type'].isin([type])]['Variable'])
    
    num_na_values = int(df_final[df_final.columns.intersection(list_vars)].isna().sum().sum())
    
    print(f"{key:>8}  {type:20}   {num_na_values:>7}")

#There are no 'duplicate' data types

#VARIABLE TYPE:        continuous
#IMPUTATION METHOD:    mean

for var in list(df_types[df_types['Insert_type']=='continuous']['Variable']):
    
    num_np_nan = df_final[var].isna().sum()
    
    value_fill = round(np.mean([x for x in df_final[var] if x >= 0]),3)
    
    if value_fill < 0 and value_fill != -2:
        
        value_fill = -2
        
    df_final[var].fillna(value_fill, inplace = True)

#VARIABLE TYPE:        discrete continuous
#IMPUTATION METHOD:    mean rounded

for var in list(df_types[df_types['Insert_type']=='discrete continuous']['Variable']):
    
    num_np_nan = df_final[var].isna().sum()
    
    value_fill = int(round(np.mean([x for x in df_final[var] if x >= 0]),0))
    
    if value_fill < 0 and value_fill != -2:
        
        value_fill = -2
        
    df_final[var].fillna(value_fill, inplace = True)

#VARIABLE TYPE:        ordered categorical
#IMPUTATION METHOD:    mode

for var in list(df_types[df_types['Insert_type']=='ordered categorical']['Variable']):
    
    num_np_nan = df_final[var].isna().sum()
    
    value_fill = int(max(statistics.multimode([x for x in df_final[var] if x >= 0])))
    
    if value_fill < 0 and value_fill != -2:
        
        value_fill = -2
        
    df_final[var].fillna(value_fill, inplace = True)

#VARIABLE TYPE:        drop

df_final.drop(columns = list(df_types[df_types['Insert_type']=='drop']['Variable']), inplace = True)

#How many 'np.nan' values for each variable type?

print(f"{'KEY':>8}  {'TYPE':20}   {'N_np.nan':>7}\n")

for key,type in dict_types.items():
    
    list_vars = list(df_types[df_types['Insert_type'].isin([type])]['Variable'])
    
    num_na_values = int(df_final[df_final.columns.intersection(list_vars)].isna().sum().sum())
    
    print(f"{key:>8}  {type:20}   {num_na_values:>7}")

ratio_twos = df_final.eq(-2).sum().sum() / (len(df_final) * len(list(df_final))) * 100
print(f"df_final: {ratio_twos} % (ratio_twos)\n")

ratio_negatives = df_final.lt(0).sum().sum() / (len(df_final) * len(list(df_final))) * 100
print(f"df_final: {ratio_negatives} % (ratio_negatives)\n")

print(df_final[(df_final < 0) & (df_final != -2)].lt(0).sum().sum())

#Are there any unavailable variables?

print(f'{set(list_protec_vars) - set(df_final[df_final.columns[df_final.columns.isin(list_protec_vars)]].columns)}')

df_final.reset_index(drop = True, inplace = True)

##############################################
##AGE**2, ADJUSTED INCOMED and WORKING HOURS##
##############################################

#VARIABLE: agesquared
#FORMULA: dvage^2
df_final['agesquared'] = np.power(df_final['d11101'],2)

#VARIABLE: lnhhinc
#FORMULA: log((1+fihhmnnet1_dv)/sqrt(hhsize))

df_final['lnhhinc'] = np.log((1 + df_final['i11102']) / np.sqrt(df_final['d11106']))

#VARIABLE: pequiv/e11101 (Annual Work Hours of Individual)
#FORMULA: e11101 / 52 weeks in a year

df_final['e11101'] = df_final['e11101'] / 52


ratio_twos = df_final.eq(-2).sum().sum() / (len(df_final) * len(list(df_final))) * 100
print(f"df_final: {ratio_twos} % (ratio_twos)\n")

ratio_negatives = df_final.lt(0).sum().sum() / (len(df_final) * len(list(df_final))) * 100
print(f"df_final: {ratio_negatives} % (ratio_negatives)\n")

print(df_final[(df_final < 0) & (df_final != -2)].lt(0).sum().sum())

dict_protec_variables_papername = {'plh0182':'lsat',
                                   'd11102ll':'female',
                                   'd11101':'age',
                                   'd11106':'numhh',
                                   'd11107':'numchildren',
                                   'd11109':'numeduc',
                                   'i11102':'hhinc',
                                   'm11124':'disabled',
                                   'm11127':'doctorvisits',
                                   'd11104':'maritalstat',
                                   'l11101_ew':'state', 
                                   'bmi':'bmi',
                                   'e11102':'empstat',
                                   'e11101':'workhours',
                                   'hlf0001_h':'homeowner',
                                   'pmonin':'month',
                                   'plh0258_h':'religion',
                                   'migback':'ethnicity'}

df_final.rename(columns = dict_protec_variables_papername, inplace = True)

df_final.reset_index(drop = True, inplace = True)

#List of all the variables in 'df_ukhls' (for CSV export)

list_var = []
list_lab = []

for i,var in enumerate(list(df_final)):
    
    list_var.append(var)
    
    try:
        
        list_lab.append(df_lookup[df_lookup['Variable'] == var]['Label'].values[0])
        
    except:
        
        list_lab.append("< PROTECTED VARIABLE >")

df_final_variables = (pd.DataFrame([list_var, list_lab]).T).rename(columns = {0:"Variable",1:"Label"})

save_here = f"{path}/05_SOEP_KS_final_variable_list.xlsx"

df_final_variables.to_excel(save_here)

print(f"list of variables exported to '{save_here}")

list_variable_list_paper_new = ['lsat',
                                'female',
                                'age',
                                'agesquared',
                                'numhh',
                                'numchildren',
                                'numeduc', # education
                                'hhinc',
                                'lnhhinc',
                                'disabled',
                                'doctorvisits', # health
                                'maritalstat',
                                'state',
                                'bmi',
                                'empstat',
                                'workhours',
                                'homeowner',
                                'month',
                                'religion',
                                'ethnicity']

print(len(list_variable_list_paper_new))

#Of the protected variables, how many values are missing 
#(i.e., nan) or 'na' (i.e., value '-2')?

(pd.DataFrame(np.array([list_variable_list_paper_new,
                        df_final[list_variable_list_paper_new].isna().sum(),
                        round(df_final[list_variable_list_paper_new].isna().sum() / len(df_final) * 100, 2),
                        df_final[list_variable_list_paper_new].eq(-2).sum(),
                        round(df_final[list_variable_list_paper_new].eq(-2).sum() / len(df_final) * 100, 2)])).T).rename(columns = {0:'protected_variable',
                                                                                                                                    1:'NaN_count',
                                                                                                                                    2:'NaN_share%',
                                                                                                                                    3:"NA_'-2'_count",
                                                                                                                                    4:"NA_'-2'_share%"})

#For each variable, how many values are missing (i.e., nan) or 'na' (i.e., value '-8')?

df_imputation = (pd.DataFrame(np.array([list(df_final),
                        df_final[list(df_final)].isna().sum(),
                        round(df_final[list(df_final)].isna().sum() / len(df_final) * 100, 2),
                        df_final[list(df_final)].eq(-2).sum(),
                        round(df_final[list(df_final)].eq(-2).sum() / len(df_final) * 100, 2)])).T).rename(columns = {0:'variable',
                                                                                                                            1:'NaN_count',
                                                                                                                            2:'NaN_share%',
                                                                                                                            3:"NA_'-2'_count",
                                                                                                                            4:"NA_'-2'_share%"})

save_here = f"{path}/06_SOEP_KS_cont_vars_imputation.csv"
df_imputation.to_csv(f"{save_here}")
print(f" >> Export completed! File exported to '{save_here}'")

df_continuous = pd.read_excel(f"{path}/07_SOEP_KS_cont_vars_imputation_filled.xlsx").drop(columns=['N'])

list_vars_to_impute_zero = df_continuous[df_continuous['imputation'] == 0]['variable']

print(f"Continuous variables to be zero-imputed: {len(list_vars_to_impute_zero)}")

for var in list_vars_to_impute_zero:
    
    df_final[var].replace(-2,0, inplace = True)

list_vars_to_impute_mean = df_continuous[df_continuous['imputation'] == 'mean']['variable']

print(f"Continuous variables to be mean-imputed: {len(list_vars_to_impute_mean)}")

for var in list_vars_to_impute_mean:
    
    df_final[var].replace(-2,df_final[var].mean(), inplace = True)

df_final

#Exporting to .csv

df_final.to_csv(f"{path}/SOEP_KS_dataset_clean.csv")


#Exporting to .dta

df_final.to_stata(f"{path}/SOEP_KS_dataset_clean.dta", variable_labels = dict(zip(df_final_variables['Variable'], df_final_variables['Label'])), write_index = False)


