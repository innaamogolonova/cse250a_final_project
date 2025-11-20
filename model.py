import numpy as np
import matplotlib as plt
import pandas as pd


model_path = "speeddating.csv"
columns_needed = ["attractive", 'attractive_partner', 'attractive_o', 'attractive_important', 'pref_o_attractive',
                  'sincere', 'sincere_partner', 'sinsere_o', 'sincere_important' , 'pref_o_sincere',
                  'intelligence', 'intelligence_partner' , 'intelligence_o', 'intellicence_important', 'pref_o_intelligence', 'match']

speed_dating_ds = pd.read_csv(model_path, usecols=columns_needed)

#print(speed_dating_ds.isnull().values.any())
#NO NULL VALUES for our dataset

print(speed_dating_ds.head(5)) #check

