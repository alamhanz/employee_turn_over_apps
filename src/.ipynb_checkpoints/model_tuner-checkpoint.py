import xgboost as xgb
import time
import numpy as np

class xgb_tuning():
    def __init__(self,obj,features,target,nthread=4):
        self.model_name = 'xgboost'
        self.metrics = {'aucpr'}
        self.obj = obj
        self.X = features
        self.y = target
        self.nthread = nthread
        self.num_boost = []
        
    def xgb_train_eval(self,max_depth,learning_rate,
                 gamma, reg_alpha,reg_lambda,n_estimators=1000,
                 kfold = True,verbose = -1):
    
        X = self.X
        y = self.y
        
        L1 = time.time()
        md= int(max_depth)
        lr= max(learning_rate,0)
        ne= int(n_estimators)
        gamma= max(gamma, 0) # 0
        ra= max(reg_alpha, 0) # 0
        rl= max(reg_lambda, 0) # 1
        see= 123
        
        N_FOLDS = 4
        STOP_ROUNDS = 10

        dtrain = xgb.DMatrix(X, label=y, nthread=self.nthread)

        param_hyp = {'max_depth' : md,
                 'eta' : lr,  # Learning Rate
                 'gamma' : gamma,
                 # 'num_parallel_tree' : ne,
                 'alpha' : ra,
                 'lambda' : rl
                 }

        if kfold :
            cv_results = xgb.cv(
                    params = param_hyp,
                    dtrain = dtrain,
                    num_boost_round=ne, # n_estimators
                    seed=see,
                    nfold=N_FOLDS,
                    metrics=self.metrics,
                    early_stopping_rounds=STOP_ROUNDS,
                    obj = self.obj
                    )

            print('Done in ',round((time.time()-L1)/60,2),'minutes')
            print(len(cv_results['test-rmse-mean']))
            
            metric_result = -1*cv_results['test-rmse-mean'].min()
            self.num_boost.append([len(cv_results['test-rmse-mean']),metric_result])
            
            return metric_result
        
        else:
            self.opt_index = np.argmax(np.array(self.num_boost)[:,1])
            self.ne_train = self.num_boost[self.opt_index][0]
            
            xgB_model = xgb.train(params = param_hyp,
                    dtrain = dtrain,
                    num_boost_round=self.ne_train, # n_estimators
                    # seed=see,
                    # metrics=self.metrics,
                    # early_stopping_rounds=STOP_ROUNDS,
                    obj = self.obj)
            return xgB_model

# class lgb_tuning():
#     def __init__(self) :
#         self.num_boost = []
        
#     def lgb_evaluate(self,max_depth = 5
#                      ,learning_rate = 0.002
#                      ,reg_alpha = 0.1
#                      ,reg_lambda = 0.1
#                      ,colsample_bytree = 0.95
#                      ,bagging_fraction = 0.95
#                      ,num_leaves = 10
#                      ,min_data = 5
#                      ,max_bin = 50
#                      ,bagging_freq = 15
#                      ,num_boost_round = 8500
#                      ,X=X_tr2,y=y_tr
#                      ,kfold = True,verbose = -1):

#     #     ra= max(reg_alpha, 0) # 0
#     #     rl= max(reg_lambda, 0) # 1

#         # Parameters
#         N_FOLDS = 4
#         MAX_BOOST_ROUNDS = int(num_boost_round) ## --> n_estimators
#         LEARNING_RATE = max(learning_rate,0)

#         params = {}
#         params['learning_rate'] = LEARNING_RATE # shrinkage_rate
#         params['boosting_type'] = 'gbdt'
#         params['objective'] = 'binary'
#         params['metric'] = ['auc','average_precision']
#         params['scale_pos_weight'] = 8

#         params['sub_feature'] = max(colsample_bytree,0.3)      # feature_fraction 
#         params['reg_alpha'] = reg_alpha
#         params['reg_lambda'] = reg_lambda
#         params['max_depth'] = int(max_depth)
#         params['bagging_fraction'] = bagging_fraction # sub_row --> same as 'subsample'
#         params['bagging_freq'] = int(bagging_freq)
#         params['num_leaves'] = int(num_leaves)        # num_leaf --> same as 'max_leaves'
#         params['min_data'] = int(min_data)         # the larger the more regulate
#         params['max_bin'] = int(max_bin) ##small number deal with overfit
#         params['min_hessian'] = 0.3     # min_sum_hessian_in_leaf

#         params['verbose'] = verbose
#         params['n_jobs'] = 25

#         d_train = lgb.Dataset(X, label=y, 
# #                               categorical_feature=col_cat,
#                               free_raw_data=False)

#         if kfold :
#             cv_results = lgb.cv(params, d_train, num_boost_round=MAX_BOOST_ROUNDS, nfold=N_FOLDS, 
#                             verbose_eval=0,early_stopping_rounds=8)
#             metric_result = cv_results['average_precision-mean'][-1]
#             self.num_boost.append([len(cv_results['auc-mean']),metric_result])
#             print(len(cv_results['auc-mean']))
#             return metric_result

#         else:
#             lgB_model = lgb.train(params, d_train, num_boost_round = MAX_BOOST_ROUNDS, verbose_eval = 250,early_stopping_rounds=5
#                          ,valid_sets=[d_train])
#             return lgB_model
        
#     def opt_numb_boost(self):
#         self.opt_index = np.argmax(np.array(self.num_boost)[:,1])
#         return lgb_tune.num_boost[self.opt_index][0]