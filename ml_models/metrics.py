from mod_man_utils import add_module
add_module("model-scout")

from model_scout.metrics import *
# from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,r2_score,mean_squared_error,mean_absolute_error
# from scipy.stats import spearmanr
# import numpy as np

# def compute_metrics(task,y_true,y_pred,y_prob=None):
#     if task=="classification":
#         m={"accuracy":float(accuracy_score(y_true,y_pred)),"f1":float(f1_score(y_true,y_pred,average="weighted"))}
#         if y_prob is not None and y_prob.shape[1]==2: m["roc_auc"]=float(roc_auc_score(y_true,y_prob[:,1]))
#         return m
#     else:
#         # Compute Spearman correlation and p-value
#         spearman_rho, spearman_p = spearmanr(y_true, y_pred)
        
#         return {
#             "r2": float(r2_score(y_true,y_pred)),
#             "rmse": float(np.sqrt(mean_squared_error(y_true,y_pred))),
#             "mae": float(mean_absolute_error(y_true,y_pred)),
#             "spearman_rho": float(spearman_rho),
#             "spearman_p": float(spearman_p)
#         }