#############################################
###MISCELLANEA: UNTANGLING ISSUES WITH OLS###
#############################################

#COMMENTS:

#This is the R script regarding the Extended Set 
#to check for multicollinearity issues in the OLS estimates in 
#"Machine Learning in the Prediction of Human Wellbeing", joint first 
#authorship by Oparina, E., Kaiser, C., and Gentile, N., and et al..

#The peculiarity of running Linear Regressions here in R using 
#the built-in lm() function is that it tells you for which
#variables it wasn't possible to estimate the coefficients.

#In this script, this issue is checked for OLS regression on the
#Extended set and the Panel dataset.

#In the Post-LASSO Extended set, we observed no 
#multicollinearity nor numerical issues when running the OLS,
#clearly due to the correlated variables being
#eliminated ex-ante in the previous phase by the
#Norm 1 LASSO penalty.

#As we are looking for multicollinearity issues, importing the
#training set is sufficient.


library(dplyr)

options(scipen=999)

read_path <- "C:\\Some\\Local\\Path\\"

##############################
##OLS ISSUES ON EXTENDED SET##
##############################

train_ks <- read.csv(paste0(read_path, 'train_ks_stand.csv')) 

#To run linregs in R, all variables (the depvar and all the
#indepvars) must be in the same dataframe.
  
###########################
##TRAIN LINEAR REGRESSION##
###########################

train_ks <- subset(train_ks, select = -c(pid))
  
lm_ks_train <- lm(lsat ~ . , data = train_ks)
  
summary(lm_ks_train) 
  
coefs_ks <- coef(lm_ks_train)
  
nas_coef_ks <- coefs[is.na(coefs_ks)]

length(nas_coef_ks)

#22
  
print(nas_coef)

#Variables to clear:
  
#i11112         m11101         m11122         m11123   plb0097_.2.0 
#NA             NA             NA             NA       NA 

#hlf0011_h_nan  ple0004_2.0   ple0164_.2.0    plb0040_nan   plj0022_.2.0 
#NA             NA            NA              NA            NA 

#e11103_2.0  plb0022_h_5.0  plb0022_h_9.0    plb0035_nan plj0116_h_.2.0 
#NA          NA             NA               NA          NA 

#e11106_nan   plb0041_.2.0   hlf0073_.2.0 plb0031_h_.2.0  hlf0092_h_nan 
#NA           NA             NA           NA              NA 

#plb0103_.2.0 plb0156_v1_nan 
#NA           NA

