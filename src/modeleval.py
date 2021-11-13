
from sklearn.metrics import mean_squared_error, make_scorer,r2_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, plot_precision_recall_curve, average_precision_score
import seaborn as sns

def binary_eval(y_true, y_pred = None, model = None, predictor = None):
    """
    [TODO] To output evaluation of binary model
    
    Args:
        predictor (pandas): 
        model (string) : column name for target value
        predictor (array of string) : name of features to be calculated
        
    Returns:
        auc_pr, auc_roc
    
    """
    
    y_val = y_true.copy()
    
    if model is not None :
        if predictor is not None :
            y_val_pred = model.predict(predictor)
        else:
            print('Insert Data for Model')
            return
    elif y_pred is not None :
        y_val_pred = y_pred.copy()
    else:
        print('Insert Model or Target Prediction')   
        return 
    
    y_val_pred2 = y_val_pred.reshape(-1)
    y_val_label = (y_val_pred2>0.5).astype(int)

    print(classification_report(y_val,y_val_label))
    
    cm = confusion_matrix(y_val,y_val_label)
    sns.heatmap((cm.transpose()/cm.sum(axis = 1)).transpose(), annot = True)

    auc_pr_val = round(average_precision_score(y_val,y_val_pred2),4)
    auc_roc_val = round(roc_auc_score(y_val,y_val_pred2),4)

    print('aucpr : ',auc_pr_val)
    print('aucroc : ',auc_roc_val)
    
    return auc_pr_val, auc_roc_val