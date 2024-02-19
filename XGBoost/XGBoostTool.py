from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
# Plotting libraries  
import matplotlib
matplotlib.use('Agg') # Bypass the need to install Tkinter GUI framework
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,log_loss

def SaveModel(input_model,name ='default'):
    try:
        input_model.model.save_model("%s.json"%name)
    except:
        print('WARNING: the model is not saved')

# Metrics: Returns [AUC, log_loss, f1] for a given model
def Metrics(model,X_test,y_test):
    print("[XGBoostTool Metrics]  Calculates logloss, roc and f1 with y_pred and y_test")
    y_pred = model.predict(X_test)
    log_loss_t =log_loss(y_test,model.predict_proba(X_test)[:, 1])
    return [roc_auc_score(y_test, y_pred),log_loss_t,f1_score(y_test, y_pred)]

def EvaluationPlot(eval_result,metric,name):
    fig, ax = plt.subplots()
    for fold, results in enumerate(eval_result):
        epochs = len(results['validation_0'][metric])
        x_axis = range(0, epochs)
        # plot log loss        
        ax.plot(x_axis, results['validation_0'][metric], label='Train_%i'%fold)
        ax.plot(x_axis, results['validation_1'][metric], label='Test_%i'%fold)
        ax.legend()
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.title('XGBoost %s'%metric)
    #plt.savefig('%s.png'%(name))
    plt.savefig('%s.pdf'%(name))
    plt.close()

def Fitter(input_model,bool_skf = False, save = True,name='', to_csv = True):
    print("[XGBoostTool Fitter]  Function to fit the model")
    df_optimization = pd.DataFrame(columns=['roc_auc','log_loss','f1','k'])
    binomial = False
    for i in range(len(input_model.X_train)):
        if len(np.unique(input_model.y_train[i]))==2:
            eval_metrics = ['logloss','auc']
            binomial = True
        else:
            eval_metrics = ['mlogloss']
        if i>0: input_model.RestartModel()
        results_cv = []
        if bool_skf:
            skf = StratifiedKFold(n_splits=5)
            print('Signal/Total: %i/%i(%.3f)'%(input_model.y_train[i].sum(),len(input_model.y_train[i]),input_model.y_train[i].sum()/len(input_model.y_train[i])))
            for train, test in skf.split(input_model.X_train[i], input_model.y_train[i]):
                input_model.RestartModel()
                print('Signal/Total: %i/%i(%.3f)(%.3f)'%(input_model.y_train[i][train].sum(),len(input_model.y_train[i][train]),input_model.y_train[i][train].sum()/len(input_model.y_train[i][train]),input_model.weigth_train[i][train].sum()))
                eval_set = [(input_model.X_train[i][train],input_model.y_train[i][train]),(input_model.X_train[i][test],input_model.y_train[i][test])]
                print("[XGBoostTool Fitter]  Fitting (bool_skf)")
                input_model.model.fit(input_model.X_train[i][train],input_model.y_train[i][train], eval_set=eval_set,eval_metric= eval_metrics,verbose=True,sample_weight=input_model.weigth_train[i][train])
                if binomial:
                    m=Metrics(input_model.model,input_model.X_train[i][test],input_model.y_train[i][test])
                    m.append(i)
                    df_optimization.loc[len(df_optimization)]=m
                results_cv.append(input_model.model.evals_result())
        else:
            eval_set = [(input_model.X_train[i],input_model.y_train[i]),(input_model.X_test[i],input_model.y_test[i])]
            print("[XGBoostTool Fitter]  Fitting (not bool_skf)")
            input_model.model.fit(input_model.X_train[i], input_model.y_train[i],eval_set=eval_set,eval_metric=eval_metrics,verbose=True,sample_weight=input_model.weigth_train[i],early_stopping_rounds=50)
            if binomial:
                m=Metrics(input_model.model,input_model.X_test[i],input_model.y_test[i])
                m.append(i)
                df_optimization.loc[len(df_optimization)]=m
            results_cv.append(input_model.model.evals_result())
        if name == '':
            for metric in eval_metrics:
                EvaluationPlot(results_cv,metric,'%s_%s_K%i'%(metric,input_model.signal,i))
            #EvaluationPlot(results_cv,'auc','auc_%s_K%i'%(input_model.signal,i))
        else:
            for metric in eval_metrics:
                EvaluationPlot(results_cv,metric,'%s_%s_K%i_%s'%(metric,input_model.signal,i,name))
            #EvaluationPlot(results_cv,'logloss','logloss_%s_K%i_%s'%(input_model.signal,i,name))
            #EvaluationPlot(results_cv,'auc','auc_%s_K%i_%s'%(input_model.signal,i,name))
        if save:
            if name == '':
                SaveModel(input_model,"model_%s_K%i"%(input_model.signal,i))
            else:
                SaveModel(input_model,"model_%s_K%i_%s"%(input_model.signal,i,name))
        input_model.AppendModel()
    if to_csv:
        if name == '':
            df_optimization.to_csv("metric_for_fold_%s.csv"%input_model.signal)
        else:
            df_optimization.to_csv("metric_for_fold_%s_%s.csv"%(input_model.signal,name))
    return df_optimization

def FeatureImportance(input_model, graphic = True, name ='',label={}):
    print("[XGBoostTool FeatureImportance]   Evalutaing the importance")
    #Remove the variables weight_nominal, eventNumber and S/B due to they are not fetures for the BDT
    #BDT_var = input_model.BDT_var
    BDT_var = []
    for var in input_model.BDT_var:
        try:
            BDT_var.append(label[var])
        except:
            print('WARNING:variable without label')
            BDT_var.append(var)
    print (BDT_var)
    df_feature_importance = pd.DataFrame({'BDT_var': BDT_var, 'importance':input_model.model.feature_importances_})
    df_feature_importance.sort_values(by ='importance',inplace= True)
    if graphic:
        # Feature importance diagram
        #axis.ticks
        plt.axes().set_yticks(np.arange(len(BDT_var)))
        plt.axes().set_yticklabels(df_feature_importance['BDT_var'])
        # axis ticks label rotation
        plt.setp(plt.axes().get_yticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
        plt.axes().set_ylim([-0.5, len(BDT_var) + 0.5])
        # resize plot
        plt.tight_layout()

        #histogram
        plt.barh(range(len(df_feature_importance['importance'])),df_feature_importance['importance'])#get_booster().get_score(importance_type= "gain"))
        #save and close
        if name=='':
            #plt.savefig('Feature_importance_%s.png'%input_model.signal)
            plt.savefig('Feature_importance_%s.pdf'%input_model.signal)
        else:
            #plt.savefig('Feature_importance_%s_%s.png'%(input_model.signal,name))
            plt.savefig('Feature_importance_%s_%s.pdf'%(input_model.signal,name))
        plt.close()

    return df_feature_importance

def OptimizeFeature(input_model, name = ''):
    print("[XGBoostTool OptimizeFeature]  Building variable ranking for a given model")
    num_var = range(len(input_model.BDT_var))
    df_optimization = pd.DataFrame(columns=['Removed','roc_auc','log_loss','f1','accuracy'])
    for i in num_var:
        if i != 0:
            input_model.RestartModel()
            input_model.ModelStored = []
            input_model.BDT_var.remove(df.iloc[0]['BDT_var'])
            input_model.data.drop(columns=df.iloc[0]['BDT_var'],inplace = True)
            input_model.ModuleSplit()
        Fitter(input_model,bool_skf=False,save = False)
        df = FeatureImportance(input_model,graphic=False)
        y_pred = input_model.model.predict(input_model.X_test[0])
        log_loss_t =log_loss(input_model.y_test[0],input_model.model.predict_proba(input_model.X_test[0])[:, 1])
        df_optimization.loc[len(df_optimization)]=[df.iloc[0]['BDT_var'],roc_auc_score(input_model.y_test[0], y_pred),log_loss_t,f1_score(input_model.y_test[0], y_pred),accuracy_score(input_model.y_test[0], y_pred)]
    try:
        if name == '':
            df_optimization.to_csv("feature_optimize_%s.csv"%input_model.signal)
        else:
            df_optimization.to_csv("feature_optimize_%s_%s.csv"%(name,input_model.signal))
    except:
        print('WARNING: feature_optimize does not save')
    return df_optimization





