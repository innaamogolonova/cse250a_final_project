import numpy as np
import matplotlib as plt
import pandas as pd


model_path = "speeddating.csv"
columns_needed = ["d_attractive", 'd_attractive_partner', 'd_attractive_o', 'd_attractive_important', 'd_pref_o_attractive',
                  'd_sincere', 'd_sincere_partner', 'd_sinsere_o', 'd_sincere_important' , 'd_pref_o_sincere',
                  'd_intelligence', 'd_intelligence_partner' , 'd_intelligence_o', 'd_intellicence_important', 'd_pref_o_intelligence', 'match']

speed_dating_ds = pd.read_csv(model_path, usecols=columns_needed)

#print(speed_dating_ds.isnull().values.any())
#NO NULL VALUES for our dataset

#print(speed_dating_ds.head(5))
columns = speed_dating_ds.columns

#Looking at unique values per column 
for col in columns:
    print(col + " number unique values:")
    print(speed_dating_ds[col].nunique())

#RESULTS 
#3 distinct values per column + Match is binary variable 

def e_step():
    #Estimate P(Z|X, Y)
    return


def m_step():
    return

def EM_training():
    #calling e_step and m_step for a certain amount of iterations
    #OR until the improvement to the CPT is under a certain threshold

    #return final CPTS
    return 

def predict_second_date():
    #takes in the CPTS
    #return the probability of 2nd date
    return 0

cpts = EM_training()
predict_second_date()
