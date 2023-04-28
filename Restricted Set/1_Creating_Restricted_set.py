############################################
###FIRST SCRIPT - CREATING RESTRICTED SET###
############################################

import os
import pandas as pd
import numpy as np
import pyreadstat


pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

pd.options.display.float_format = '{:.4f}'.format

'''

COMMENTS:

This is the first script of the Restricted set producing the results in 
"Human Wellbeing and Machine Learning" by Ekaterina Oparina* (r) Caspar Kaiser* (r) Niccolò Gentile*; 
Alexandre Tkatchenko, Andrew E. Clark, Jan-Emmanuel De Neve 
and Conchita D'Ambrosio (* equal first authors in random order).

In this one, in particular, the aim is to recreate the Restricted set of variables 
in SOEP. Scripts for are Gallup and UKHLS are similar.

In this script, we simply start importing the variables from the SOEP
repository.

Tha variables are imported from five datasets, namely:
pequiv, health, pl, ppath, and hl. We remind the reader to the relevant
SOEP documentation for further inspections.

Coherence in the observed data is guranteed by joining 
the individual identifiers (ids) across the different datasets. 
'''


path_1 = 'C:\\Here\\Some\\Local\\Path\\'

vars_pequiv = ['pid', 'syear', 'd11102ll', 'd11101', 'd11106', 'd11107','d11109','i11102']

vars_pequiv_added = vars_pequiv + ['m11124', 'm11127', 'd11104', 'l11101', 'e11102', 'e11101']

pequiv_obj = pyreadstat.read_dta(path_1 + 'pequiv.dta', usecols = vars_pequiv_added)

pequiv = pequiv_obj[0]

#Renaming some variables

pequiv.rename(columns = {'pid' : 'pid', 
                         'syear' : 'year', 
                         'd11102ll' : 'Gender',
                         'd11101' : 'Age',
                         'd11106' : 'Number of people in the household',
                         'd11107' : 'Number of children in the hh', 
                         'd11109': 'Number of years of education',
                         'i11102': 'Household yearly disposable income', 
                         'm11124' : 'Disability Status', 
                         'm11127' : 'Number of doctor visits in previous year',
                         'd11104' : 'Marital Status',  
                         'l11101': 'State of Residence',
                         'e11102': 'Employment status dummies',
                         'e11101': 'Working hours'
                         }, inplace = True)

del pequiv_obj, vars_pequiv, vars_pequiv_added

vars_health = ['pid', 'syear', 'bmi']

health_obj = pyreadstat.read_dta(path_1 + '\\health.dta', usecols = vars_health)

health = health_obj[0]

health.rename(columns = {'pid' : 'pid', 
                         'syear' : 'year', 
                         'bmi' : 'BMI'}, inplace = True)

del health_obj, vars_health

cols_pl = ['pid', 'syear', 'plh0182', 'pmonin', 'plh0258_h', 'hid']

pl_obj = pyreadstat.read_dta(path_1 + '\\pl.dta', usecols = cols_pl)

pl = pl_obj[0]

pl.rename(columns = {'pid' : 'pid', 
                     'syear' : 'year',
                     'plh0182' : 'Life Satisfaction',
                     'pmonin': 'Month of Interview',
                     'plh0258_h': 'Religion',
                     'hid': 'hid'}, inplace = True)

del pl_obj, cols_pl

cols_hl = ['hid', 'syear', 'hlf0001_h']

hl_obj = pyreadstat.read_dta(path_1 + '\\hl.dta', usecols = cols_hl)

hl = hl_obj[0]

#For house ownership, hlf0001_h.

#Details about the considered variables can be found at:

#https://paneldata.org/search/all

hl.rename(columns = {'hid' : 'hid', 
                     'syear' : 'year',
                     'hlf0001_h': 'Housing ownership status'}, inplace = True)

del hl_obj, cols_hl

dset_1 = pequiv.merge(health, on = ['pid', 'year'], how = 'inner')

dset_2 = dset_1.merge(pl, on = ['pid', 'year'], how = 'inner')

dset_2.shape

#681699 x 20

#As specified in the paper, for for the Restricted Set we consider
#the data ony from 2010 to 2018 included.

dset_2 = dset_2[dset_2['year'] > 2009]

dset_2.shape

#260357 x 20

#print(dset_2.head(25))

hl = hl[hl['year'] > 2009]

#First: are all the hids in hl also available in dset_2?

len(list(set(hl['hid']) - set(dset_2['hid'])))

#5 Only are different.

#Let's drop those hid that are not in common.

len(hl) #155020

hl = hl[hl['hid'].isin(dset_2['hid'])]

len(hl) #155015

#Now, both in hl and pl we have the same hid. Let's sort the 
#rows based on them.

hl.sort_values(by = ['hid', 'year'], inplace = True)

dset_2.sort_values(by = ['hid', 'year', 'pid'], inplace = True)

dset_3 = dset_2.merge(hl, on = ['hid', 'year'], how = 'inner')


#Migback is here:

vars_ppath = ['pid', 'migback']

ppath_obj = pyreadstat.read_dta(path_1 + 'ppath.dta', usecols = vars_ppath)

ppath = ppath_obj[0]

ppath.rename(columns = {'pid': 'pid',
                        'migback': 'ethnicity'}, inplace = True)


ppath.sort_values(by = ['pid'], inplace = True)

dset_3.sort_values(by = ['pid'], inplace = True)

dset = dset_3.merge(ppath, on = ['pid'], how = 'inner')

dset.to_csv('C:\\Some\\Local\\Path\\Name_of_temporary_dset_1.csv')

