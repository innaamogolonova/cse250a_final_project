import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_features(f="all",binned=False):
    attractive = ["attractive","attractive_partner",  "attractive_o",  "attractive_important",  "pref_o_attractive"]
    sincere = ["sincere",      "sincere_partner",     "sinsere_o",     "sincere_important",     "pref_o_sincere"]
    intel = ["intelligence",   "intelligence_partner","intelligence_o","intellicence_important","pref_o_intelligence"]
    
    d_attractive = ["d_attractive","d_attractive_partner",  "d_attractive_o",  "d_attractive_important",  "d_pref_o_attractive"]
    d_sincere = ["d_sincere",      "d_sincere_partner",     "d_sincere_o",     "d_sincere_important",     "d_pref_o_sincere"]
    d_intel = ["d_intelligence",   "d_intelligence_partner","d_intelligence_o","d_intelligence_important","d_pref_o_intelligence"]

    features = attractive + sincere + intel
    d_features = d_attractive + d_sincere + d_intel

    if f == "attractive":
        if binned:
            return d_attractive
        else:
            return attractive
    elif f == "sincere":
        if binned:
            return d_sincere
        else:
            return sincere
    elif f == "intelligence":
        if binned:
            return d_intel
        else:
            return intel
    else:
        if binned:
            return d_features
        else:
            return features


def get_feature_df():
    orig_df = pd.read_csv("../speeddating.csv",low_memory=False)

    kept_features = get_features() + ["match"]

    feature_df = orig_df[kept_features].apply(pd.to_numeric, errors='coerce').fillna(-1).astype(int)

    return feature_df


def get_data_info(feature_df,show_plot=False):
    pred_col = feature_df["match"]

    print(feature_df.info(verbose = True, show_counts = True))
    print(pred_col.value_counts(normalize=True) * 100)
    if show_plot:
        sns.displot(feature_df, x="match")
        plt.show()



def get_col_prob_dict(df,col_name):
    col_vals = df[col_name].tolist()
    total = len(col_vals)
    unique_vals = set(col_vals)
    prob_dict = {}

    for i in unique_vals:
        count = col_vals.count(i)
        prob_dict[i] = count / total
    
    return prob_dict



def get_all_cols_prob_dict(df,col_names):
    all_probs_dict = {}
    
    for i in col_names:
        prob_dict = get_col_prob_dict(df,i)
        all_probs_dict[i] = prob_dict
    
    return all_probs_dict


def get_prob_dict(feature_df,kept_features,col_name):
    all_probs_dict = get_all_cols_prob_dict(feature_df,kept_features)
    
    return all_probs_dict[col_name]








