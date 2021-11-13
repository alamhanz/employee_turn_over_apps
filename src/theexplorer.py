import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eval_features(df_input,col_label='y',col_features_selected=['1','2']):
    """
    [TODO] To output GINI score and PSI scores for each feature
    
    Args:
        df_input (pandas): 
        col_label (string) : column name for target value
        col_features_selected (array of string) : name of features to be calculated
        
    Returns:
        feat_summary
        
    """
    
    # Variables initiation
    d = dict() # for storing GINI values
    p = dict() # for storing PSI scores
    
    for c in col_features_selected: 
        x = df_input[c]
        y = df_input[col_label]
        idx = ~x.isnull()
        x_ = x[idx]
        y_ = y[idx]
        
        # Store the GINI score
        try:
            d[c]=2*roc_auc_score(y_,x_)-1
        except:
            print(c)
            d[c]=2*roc_auc_score(y_,x_)-1
        
        yval=y.unique()
        df_features1=df_input[[c]]
        df_1=df_input[df_input[col_label]==yval[0]][[c]]
        df_2=df_input[df_input[col_label]==yval[1]][[c]]
        
        # [TODO] Binning to calculate the PSI
        n=10
        bins=[]
        range_step=100/n
        steps=0
        while steps+range_step<100:
            try:
                bins.append((np.percentile(df_features1,steps),np.percentile(df_features1,steps+range_step)))
                steps=steps+range_step
            except:
                print(c)
                bins.append((np.percentile(df_features1,steps),np.percentile(df_features1,steps+range_step)))
                steps=steps+range_step
                

        df1_probs=[]
        df2_probs=[]
        
        # [TODO] Compute probability for each bin to calculate PSI
        for bin0 in bins:
            df1_probs.append(len(df_1[(df_1[c]>=bin0[0])&(df_1[c]<bin0[1])])/
                                float(len(df_1)))
            df2_probs.append(len(df_2[(df_2[c]>=bin0[0])&(df_2[c]<bin0[1])])/
                                float(len(df_2)))

        df1_probs.append(len(df_1[(df_1[c]>=bins[-1][1])])/
                            float(len(df_1)))
        df2_probs.append(len(df_2[(df_2[c]>=bins[-1][1])])/
                            float(len(df_2)))


        df_prob_dist=pd.DataFrame({'prob1':df1_probs,'prob2':df2_probs})
        
        # Calculate PSI values
    #     df_prob_dist['karawang']=df_prob_dist['karawang'].replace(0,0.0000000001)
        df_prob_dist['psi_calc1']=(df_prob_dist['prob1']-df_prob_dist['prob2'])
        df_prob_dist['psi_calc2']=(np.log(df_prob_dist['prob1']/df_prob_dist['prob2'])).fillna(0).replace(np.inf,10)
        df_prob_dist['psi']=df_prob_dist['psi_calc2']*df_prob_dist['psi_calc1']

        p[c]=df_prob_dist.psi.sum()
        
        

    gini_per_features = pd.DataFrame.from_dict(d,orient='index').rename(columns={0:'gini'})
    gini_per_features['gini_abs'] = np.abs(gini_per_features['gini'])
    
    psi_per_features = pd.DataFrame.from_dict(p,orient='index').rename(columns={0:'psi_score'})
    gini_per_features['psi_score']=psi_per_features['psi_score']
    feat_summary=gini_per_features.reset_index()
    feat_summary.columns=['features','gini','gini_abs','psi_score']
    
    return feat_summary

def plot_compare(mydata,col_target,col_x):
    d1=mydata[mydata[col_target]==1][col_x]
    d2=mydata[mydata[col_target]==0][col_x]
    plt.figure(figsize=(10,10))
    sns.distplot(d1,label='class 1')
    sns.distplot(d2,label='class 0')
    plt.legend(loc='upper left')