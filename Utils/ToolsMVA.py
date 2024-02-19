import os,sys
# Data treatment libraries  
import numpy as np
import pandas as pd
import random
from scipy import stats

import matplotlib
matplotlib.use('Agg') # Bypass the need to install Tkinter GUI framework
import matplotlib.pyplot as plt

from root_numpy import root2array, rec2array

from ROOT import TFile, TTree

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, confusion_matrix
sys.path.append('../XGBoost/')
from XGBoostTool import Fitter, Metrics

method= {'XGBoost':'XGBClassifier()',
         'NNKeras':'Sequential()'}

#from keras.models import Sequential

import array

sys.path.append('../../tHqUtils/')
from data_file_DSID import DSID_LIST

def ModuleKfold(K,df,field,BDT_var, weigth = True):

    df['Kfold']=df[field]%K

    X_train_Kfold =[]
    X_test_Kfold =[]
    y_train_Kfold = []
    y_test_Kfold = []
    weigth_train_Kfold = []
    weigth_test_Kfold=[]

    for i in range(K):
        X_train_Kfold.append(df[df['Kfold']!=i][BDT_var].to_numpy())
        X_test_Kfold.append(df[df['Kfold']==i][BDT_var].to_numpy())
        y_train_Kfold.append(df[df['Kfold']!=i]['S/B'].to_numpy()) 
        y_test_Kfold.append(df[df['Kfold']==i]['S/B'].to_numpy()) 
        if weigth:
            weigth_train_Kfold.append(df[df['Kfold']!=i]['weight_nominal'].to_numpy()) 
            weigth_test_Kfold.append(df[df['Kfold']==i]['weight_nominal'].to_numpy())
    df.drop(columns =['Kfold'],inplace = True)
    return X_train_Kfold,X_test_Kfold,y_train_Kfold,y_test_Kfold,weigth_train_Kfold,weigth_test_Kfold

def ModuleSplit(K,df,field,BDT_var, weigth = True):

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    weigth_train = []
    weigth_test = []
    
    X_train.append(df[df[field]%K!=0][BDT_var].to_numpy())
    X_test.append(df[df[field]%K==0][BDT_var].to_numpy())
    y_train.append(df[df[field]%K!=0]['S/B'].to_numpy() )
    y_test.append(df[df[field]%K==0]['S/B'].to_numpy() )
    if weigth:
        try:
            weigth_train.append(df[df[field]%K!=0]['weight_nominal'].to_numpy())
            weigth_test.append(df[df[field]%K==0]['weight_nominal'].to_numpy())
        except:
            print('WARNING: weights not found')
            #weigth_train.append(df[df[field]%K!=0]['weight_nominal'].to_numpy())
            #weigth_test.append(df[df[field]%K==0]['weight_nominal'].to_numpy())
    return X_train,X_test,y_train,y_test,weigth_train,weigth_test

def get_yield(df, cuts=''):
    import math
    lumi = 140.5
    try:
        if cuts =='':
            df_aux = df
        else:
            df_aux = df.query(cuts)
        yields = (df_aux['weight_nominal'].sum())*lumi
        raw = df_aux.shape[0]
        error_yield = math.sqrt((df_aux['weight_nominal']**2).sum())
       # print('Sum of weight: %f --> event yield %f'%(df_aux['weight_nominal'].sum(),yields))
    except:
        yields, error_yield, raw = 0,0,0
    return yields, error_yield, raw 
def ApplyPrediction(input_model, name = ''):
    df_KS_test = pd.DataFrame(columns=['k','signal','p_value_s','background','p_value_b'])
    df_result =pd.DataFrame(columns=['BDT','true_target','weight','Modul'])
    df_metrics= pd.DataFrame(columns=['roc_auc','log_loss','f1','k'])
    for i in range(len(input_model.X_train)):
        m=Metrics(input_model.ModelStored[i],input_model.X_test[i],input_model.y_test[i])
        m.append(i)
        df_metrics.loc[len(df_metrics)]=m
        
        lr_probs = input_model.ModelStored[i].predict_proba(input_model.X_test[i])[:, 1]
        lr_probs_train = input_model.ModelStored[i].predict_proba(input_model.X_train[i])[:, 1]

        lr_fpr, lr_tpr, _ = roc_curve(input_model.y_test[i], lr_probs)
        lr_fpr_train, lr_tpr_train, _ = roc_curve(input_model.y_train[i], lr_probs_train)

        array_test =  np.array([lr_probs,input_model.y_test[i],input_model.weigth_test[i]])
        df_test =pd.DataFrame(array_test)
        df_test = df_test.T
        
        array_train =  np.array([lr_probs_train,input_model.y_train[i],input_model.weigth_train[i]])
        df_train =pd.DataFrame(array_train)
        df_train = df_train.T
    
        #BDT prediction
        #Figure size
        fig, ax = plt.subplots(figsize=(8, 4))
    
        ax.hist(df_test[df_test[1]==0][0],65,  histtype='step', label='Other test',color="y",weights=(np.ones(len(df_test[df_test[1]==0][2]))/df_test[df_test[1]==0][2].sum())*df_test[df_test[1]==0][2])
        ax.hist(df_test[df_test[1]==1][0],65,  histtype='step', label='$tH$ test',color="r",weights=(np.ones(len(df_test[df_test[1]==1][2]))/df_test[df_test[1]==1][2].sum())*df_test[df_test[1]==1][2])
    
        ax.hist(df_train[df_train[1]==0][0],65,  histtype='step', label='Other train',weights=(np.ones(len(df_train[df_train[1]==0][2]))/df_train[df_train[1]==0][2].sum())*df_train[df_train[1]==0][2])
        ax.hist(df_train[df_train[1]==1][0],65,  histtype='step', label='$tH$ train',weights=(np.ones(len(df_train[df_train[1]==1][2]))/df_train[df_train[1]==1][2].sum())*df_train[df_train[1]==1][2])
    
        ax.legend(loc='upper right')
        ax.set_title('BDT prediction')
        ax.set_xlabel('BDT')
        if name == '':
            #plt.savefig("BDT_response_%s_K%i.png"%(input_model.signal,i))
            plt.savefig("BDT_response_%s_K%i.pdf"%(input_model.signal,i))
        else:
            #plt.savefig("BDT_response_%s_K%i_%s.png"%(input_model.signal,i,name))
            plt.savefig("BDT_response_%s_K%i_%s.pdf"%(input_model.signal,i,name))
        plt.close()
    
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            n_bins = 50
            # Plot the cumulative histogram for BDT prediction
            n_1, bins, patches = ax.hist(df_test[df_test[1]==1][0], n_bins, density=True, histtype='step',
                           cumulative=True, label='Test $tH$', weights= df_test[df_test[1]==1][2])
            n_3, bins, patches = ax.hist(df_test[df_test[1]==0][0], n_bins, density=True, histtype='step',
                           cumulative=True, label='Test Other', weights= df_test[df_test[1]==0][2])

            n_2, bins, patches = ax.hist(df_train[df_train[1]==1][0], n_bins, density=True, histtype='step',
                           cumulative=True, label='Train $tH$',weights= df_train[df_train[1]==1][2])
            n_4, bins, patches = ax.hist(df_train[df_train[1]==0][0], n_bins, density=True, histtype='step',
                           cumulative=True, label='Train Other', weights= df_train[df_train[1]==0][2])
        
         # tidy up the figure
            ax.grid(True)
            ax.legend(loc='upper right')
            ax.set_title('Cumulative step histograms')
            ax.set_xlabel('BDT')
            ax.set_ylabel('Likelihood of occurrence')
            if name == '':
                #plt.savefig("BDT_cumulative_response_%s_K%i.png"%(input_model.signal,i))
                plt.savefig("BDT_cumulative_response_%s_K%i.pdf"%(input_model.signal,i))
            else:
                #plt.savefig("BDT_cumulative_response_%s_K%i_%s.png"%(input_model.signal,i,name))
                plt.savefig("BDT_cumulative_response_%s_K%i_%s.pdf"%(input_model.signal,i,name))
            plt.close()
            
        #Kolmogorov-Smirnov  test
        #Documentation : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
            df_KS_test.loc[len(df_KS_test)]=[i,stats.ks_2samp(n_1,n_2)[0],stats.ks_2samp(n_1,n_2)[1],stats.ks_2samp(n_3,n_4)[0],stats.ks_2samp(n_3,n_4)[1]]
            print('SIGNAL-->K-S:%f with p-value: %f'%(stats.ks_2samp(n_1,n_2)[0],stats.ks_2samp(n_1,n_2)[1]))
            print('BKG-->K-S:%f with p-value: %f'%(stats.ks_2samp(n_3,n_4)[0],stats.ks_2samp(n_3,n_4)[1]))
        
        except:
            print('Warning K-S')
        ##################
        plt.plot(lr_fpr, lr_tpr, marker='.', markersize=0.2,label='Test_%i'%i)
        plt.plot(lr_fpr_train, lr_tpr_train, marker='.',markersize=0.2, label='Train_%i'%i)
        df_test.columns = ['BDT','true_target','weight']
        df_test['Modul']=i
        df_result = pd.concat([df_result,df_test],ignore_index = True, axis =0)
    # axis labels                                                                                                                                                                                                                                                                  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend                                                                                                                                                                                                                                                              
    plt.legend()
    #resize plot
    plt.tight_layout() 
    # save and close plot
    if name == '':
        #plt.savefig('Roc_curve_%s.png'%input_model.signal)
        plt.savefig('Roc_curve_%s.pdf'%input_model.signal)
    else:
        #plt.savefig('Roc_curve_%s_%s.png'%(input_model.signal,name))
        plt.savefig('Roc_curve_%s_%s.pdf'%(input_model.signal,name))
    plt.close()


    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})

    # Plot ROC curves on the upper panel
    ax1.plot(lr_fpr, lr_tpr, marker='.', markersize=0.2, label='Test_%i' % i)
    ax1.plot(lr_fpr_train, lr_tpr_train, marker='.', markersize=0.2, label='Train_%i' % i)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    ax1.set_title('ROC Curve')

  
    # Since lr_tpr_train and lr_tpr dont have the sane sice, we interpolate
    common_fpr = np.linspace(0, 1, 100)
    interp_tpr_test = np.interp(common_fpr, lr_fpr, lr_tpr)
    interp_tpr_train = np.interp(common_fpr, lr_fpr_train, lr_tpr_train)

    # Calculate the ratio
    ratio = interp_tpr_train / interp_tpr_test

    # Plot the ratios on the lower panel
    ax2.plot(common_fpr, ratio, marker='o', linestyle='-', label='Train/Test Ratio')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('Train/Test TPR Ratio')
    ax2.set_title('Train/Test Ratio per FPR')
    ax2.legend()
    
    # Save and close plot
    if name == '':
        plt.savefig('Roc_curve_with_ratio_%s_K%i.pdf' % (input_model.signal, i))
    else:
        plt.savefig('Roc_curve_with_ratio_%s_K%i_%s.pdf' % (input_model.signal, i, name))
    plt.close()





        ###############
    if name == '':
        df_KS_test.to_csv('Kolmogorov_Smirnov_test_%s.csv'%input_model.signal)
    else:
        df_KS_test.to_csv('Kolmogorov_Smirnov_test_%s_%s.csv'%(input_model.signal,name))
    
    if True:
        if name == '':
            df_metrics.to_csv("metric_for_fold_all_%s.csv"%input_model.signal)
        else:
            df_metrics.to_csv("metric_for_fold_all_%s_%s.csv"%(input_model.signal,name))



    return df_result






def ConfussionMatrix(input_model,name = ''):
    for i in range(len(input_model.X_test)):
        y_pred = input_model.ModelStored[i].predict(input_model.X_test[i])
        y_pred = y_pred.astype('int')
        y_test = input_model.y_test[i].astype('int')
        confusion = confusion_matrix(y_test,y_pred,sample_weight = input_model.weigth_test[i])
        column=[]
        if len(np.unique(y_test))==2:
            column = ['Signal','Background']
        else:
            for l in np.unique(y_test):
                column.append(input_model.MultiClass[l])
        df = pd.DataFrame(confusion)
        df.index =  column
        df.columns = column
        if name == '':
            df.to_csv('ConfussionMatrix_K%i.csv'%i)
        else:
            df.to_csv('ConfussionMatrix_K%i_%s.csv'%(i,name))
    return df
    
def EvalModel(input_model,param):
    print("[ToolsMVA  EvalModel]  Evaluating model with Fitter() and returning Metrics()")
    # param is a dictionary
    # input_model is df
    try:
        if param.keys()[0] in input_model.param:
            input_model.param.update(param)
        else:
            print('The parameter %s to evaluate does not exist in the model parameter'%param.keys()[0])
            return
    except:
        print("[ToolsMVA  EvalModel]  Input parameters")
        print(input_model.param)
        print('WARNING: Problem with the parameter before evaluating')
        return
    print("[ToolsMVA  EvalModel]  Restarting model")
    input_model.RestartModel()
    input_model.ModelStored = []
    #input_model.data.query('200>weight_nominal > 0',inplace=True)
    if input_model.method == 'XGBoost':
        print("[ToolsMVA  EvalModel]  Execute Fitter()")
        Fitter(input_model,bool_skf=False,save= False,to_csv = False)
    #metric = Metrics(input_model.model,input_model.X_test[0],input_model.y_test[0])
    #print(metric)
    #return(1-Metrics(input_model.model,input_model.X_test[0],input_model.y_test[0])[0])
    print("[ToolsMVA  EvalModel]  Fitter complete. RestartData and  ModuleSplit")
    input_model.RestartData()
    input_model.ModuleSplit()
    # Metrics: Returns [AUC, log_loss, f1] for a given model
    print("[ToolsMVA  EvalModel]  Metrics")
    return(Metrics(input_model.model,input_model.X_test[0],input_model.y_test[0]))

def GradientOptimize(input_model,param, learning_rate=1e-3, h=1e-3, n_evaluations=10):
    df_gradient = pd.DataFrame(columns=['metric','param','auc'])
    try:
        x_current = input_model.param[param.keys()[0]]
        print(x_current)
    except:
        print(input_model.param)
        print('param does not exist in the current model')
        return
    metrics = EvalModel(input_model,param)
    f_current = metrics[1]
    df_gradient.loc[len(df_gradient)]=[f_current,x_current,metrics[0]]

    for _ in range(n_evaluations):
        metrics = EvalModel(input_model,param)
        f_current = metrics[1]
        #f_current = EvalModel(input_model,param)
        x = x_current - h
        if x<0: x = abs(x)
        param[param.keys()[0]] = x
        metrics = EvalModel(input_model,param)
        df = f_current - metrcis[1]
        grad = df / h
        df_gradient.loc[len(df_gradient)]=[f_current,x_current,metrics[0]]

        x_current = x_current - learning_rate * grad
        param[param.keys()[0]] = x_current
        #history_x.append(x_current)
        #if abs(x-x_current)/x_current<1e-10:
        #    break
        
        if x_current < 0: break
    df_gradient.to_csv('opt_grad_%s.csv'%param.keys()[0])
    return df_gradient  

def RandomOptimize(input_model,param,min,max,n_point, name = ''):
    df_random = pd.DataFrame(columns=['%s'%param.keys()[0],'log_loss','auc'])
    try:
        x_current = input_model.param[param.keys()[0]]
        print(x_current)
    except:
        print(input_model.param)
        print('param does not exist in the current model')
        return
    metrics = EvalModel(input_model,param)
    f_current = metrics[1]
    df_random.loc[len(df_random)]=[f_current,x_current,metrics[0]]
    rnd = 10**np.random.uniform(low=min, high=max, size=(n_point,))
    for point in rnd:
        param[param.keys()[0]] = point
        metrics = EvalModel(input_model,param)
        f_current = metrics[1]
        df_random.loc[len(df_random)]=[point,f_current,metrics[0]]
    if name == '':
        df_random.to_csv('opt_rand_%s.csv'%param.keys()[0])
    else:
        df_random.to_csv('opt_rand_%s_%s.csv'%(param.keys()[0],name))
    return df_random
################################################################################################
def UpdateTree(File,filename,tree,branches,BDT_var,model,out_dir,branch_out, verbose = False):
    import time, copy
    #import multiprocessing as mp
    #pool = mp.Pool(processes = 20)
    start_time = time.time()
    dirName = out_dir
    print("UpdateTree()")
    print("UpdateTree() :: Input file: "+str(File))
    print("UpdateTree() :: output dir: " +str(out_dir))
    print branches
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    not_in_bdt_train = ['DSID', 'NEvent','weight_nominal','S/B']
    for var in not_in_bdt_train:
        try:
            branches.remove(var)
        except:
            #print('WARNING: the variable %s is not included in data'%var)
            pass
    
    try:
        print("Trying root2array")
        array_input = root2array(File,tree,branches)
        print("Trying rec2array")
        array_input = rec2array(array_input)

    except Exception as e: 
        print(e)
        print('UpdateTree() :: The file %s is not included'%File)
        return
    print("UpdateTree() :: Reading: %s seconds for file %s " % ((time.time() - start_time),File))
    apply_time = time.time()

    columns = branches

    df = pd.DataFrame(array_input, columns=columns)

    Data = False
    if filename.split('.')[1]=='periodAllYear': Data = True
    BDT_responde = []
    i = 0.
    bdt_average = 0.
    if Data:
        df['Kfold']=np.random.randint(0,5,len(df))
    else:
        df['Kfold']=df['eventNumber']%len(model)
    
    df_BDT = []
    for i  in range(len(model)):
        if len(df[df['Kfold']==i])== 0: continue
        #print("UpdateTree() :: Predicting BDT score for fold" + str(i))
        df_aux = pd.DataFrame(model[i].predict_proba(df[df['Kfold']==i][BDT_var])[:, 1], index = list(df[df['Kfold']==i].index),columns=['BDT'])
        #print("UpdateTree() :: predict_proba for fold "+str(i)+" = "+str(model[i].predict_proba(df[df['Kfold']==i][BDT_var]))) #redict_proba retrunds the probability for each class. 
        df_BDT.append(df_aux)


    df_final = pd.concat(df_BDT)
    df_final.sort_index(inplace=True)
    BDT_responde = df_final['BDT'].values
    
########################################
# OLD WAY TO EVALUATE THE BDT
########################################
    #for index, row in df.iterrows():
    #    #i = i + 1
    #    if Data:
    #        number_of_model = random.randint(0,4)
    #        weight = np.array([1.])
    #    else:
    #        try:
    #            number_of_model = int(row['eventNumber']%5)
    #        except:
    #            print('error')
    #            print(df.shape[0])
    #            print(row)
    #            print('test')
    #            print(row['m_nNoBjets'])
    #    
    #    bdt_time = time.time()
    #    if (number_of_model < len(model)):
#
    #        BDT_responde.append(model[number_of_model].predict_proba([row[BDT_var].values])[:,1][0])
    #        #pool.apply_async(model[number_of_model].predict_proba,args=([row[BDT_var].values],))
    #       
    #        #BDT_responde.append(pool.apply_async(model[number_of_model].predict_proba,args=([row[BDT_var].values],)).get()[:,1][0])
    #    else:
    #        BDT_responde.append(-999)
    #    
    #    bdt_average = bdt_average + (time.time() - bdt_time)
    #print BDT_responde
    #df_final = pd.DataFrame(BDT_responde, columns = ['BDT'])

    #df_final.to_csv('mode_by_df.csv')
    #    if verbose:
    #        i = i + 1
    #        status = float(i)/float(df.shape[0])*100
    #        sys.stdout.write('\r')
    #        sys.stdout.write("[%-100s] %.3f%%(%i/%i)" % ('='*int(status), status,i,df.shape[0]))
    #        sys.stdout.flush()
    #pool.close()
    #pool.join()
########################################
########################################
    bdt_average = bdt_average / len(df)
    #print("UpdateTree() :: Average for each prediction (n = %f) %f for file %s"% (len(df),bdt_average,File))
    print("UpdateTree() :: Applying BDT:%s seconds for file %s" % ((time.time() - apply_time),File))
    write_time = time.time()
    m_file = TFile.Open(File)
    m_tree = m_file.Get(tree)
    out_file = TFile(dirName+'/'+filename,'recreate')
    new_tree = m_tree.CloneTree()  
    #New
    m_file.Close()
    ###
    response = array.array('f', [0.0])
    branch = new_tree.Branch("bdt_%s"%branch_out, response, 'response/F')
    print('start to fill new branch')

    for i in BDT_responde:
        response[0] = i
        branch.Fill()


    print("Writting :--- %s seconds for file %s---" % ((time.time() - write_time),File))
    out_file.Write()
    out_file.Close()

def LoadModel(input_model,dir_path,tree_t,output,input_file=[],thread = False):
    print("LoadModel()")
    if thread: 
        #import threading
        #threads = []
        import multiprocessing as mp
        pool = mp.Pool(processes = 20)
    if input_file!=[]:
        input_model.ModelStored = []
        for fil in input_file:
            print("LoadModel() :: Input file: " + str(fil))
            input_model.RestartModel()
            input_model.model.load_model(fil)
            input_model.AppendModel()
    branches_t = list(input_model.data.columns.values)
    branch_name = input_model.signal
    i=0
    for file in os.listdir(dir_path):
        file_path = dir_path + file
        if os.path.splitext(file)[-1]!='.root': continue
#        try:
#            #if file.split('.')[1] not in ['512168','512169'] : continue 
#            if os.path.splitext(file)[0] not in ['ttW_Sherpa'] : continue
#        except:
#            print bcolors.WARNING + 'PROBLEM WITH  THE NAME' + bcolors.ENDC 
        #run = False
        #if tree_t.split('tHqLoop_')[-1] == 'JET_JER_DataVsMC_MC16__1down_Loose' and file == 'ttbar.root': run = True
        #if tree_t.split('tHqLoop_')[-1] == 'JET_JER_DataVsMC_MC16__1down_PseudoData_Loose' and file == 'Zjet.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_DataVsMC_MC16__1up_PseudoData_Loose' and file == 'Zjet.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_DataVsMC_MC16__1up_Loose' and file == 'Zjet.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_DataVsMC_MC16__1down_Loose' and file == 'Zjet.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'MET_SoftTrk_Scale__1down_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_1__1up_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_3__1up_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_9__1up_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_2__1up_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_1__1down_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_2__1down_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_9__1up_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_4__1down_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_5__1up_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_6__1down_PseudoData_Loose' and file == 'VBFH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'MET_SoftTrk_Scale__1down_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_PunchThrough_MC16__1down_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_SingleParticle_HighPt__1up_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_2__1up_PseudoData_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_6__1up_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'MET_SoftTrk_ResoPara_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_8__1down_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_3__1down_PseudoData_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_8__1down_PseudoData_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_7__1down_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_7__1down_PseudoData_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_5__1up_PseudoData_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_7__1up_PseudoData_Loose' and file == 'WH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_3__1up_PseudoData_Loose' and file == 'ggH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_9__1down_PseudoData_Loose' and file == 'ggH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_1__1down_PseudoData_Loose' and file == 'ggH.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_SingleParticle_HighPt__1down_Loose' and file == 's-channel.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_4__1up_PseudoData_Loose' and file == 's-channel.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_5__1down_PseudoData_Loose' and file == 's-channel.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'MET_SoftTrk_ResoPara_Loose' and file == 's-channel.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_EffectiveNP_8__1up_Loose' and file == 's-channel.root': run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_Pileup_OffsetNPV__1down_Loose' and file == 's-channel.root': run = True
        #elif (tree_t.split('tHqLoop_')[-1] == 'JET_JER_DataVsMC_MC16__1down_PseudoData_Loose' and file == 'ttbar.root'): run = True
        #elif (tree_t.split('tHqLoop_')[-1] == 'JET_JER_DataVsMC_MC16__1up_PseudoData_Loose' and file == 'ttbar.root'): run = True
        #elif (tree_t.split('tHqLoop_')[-1] == 'JET_PunchThrough_MC16__1up_Loose' and file == 'ttbar.root'): run = True
        #elif tree_t.split('tHqLoop_')[-1] == 'JET_JER_DataVsMC_MC16__1up_Loose' and file == 'ttbar.root': run = True
        #else:
        #    continue
        #if run : 
        #    print tree_t.split('tHqLoop_')[-1]
        #    print file
        #    #continue
        tree_name = tree_t
        if tree_t.split('tHqLoop_')[-1] in ['JET_PunchThrough_MC16__1down_Loose','JET_PunchThrough_MC16__1up_Loose','JET_JER_DataVsMC_MC16__1down_Loose','JET_JER_DataVsMC_MC16__1down_PseudoData_Loose','JET_JER_DataVsMC_MC16__1up_Loose','JET_JER_DataVsMC_MC16__1up_PseudoData_Loose']:
            if file.split('.')[-2] in ['tHq','tWH','SM4top','tHq_ytm1','tWH_ytm1']:
                if tree_t.split('tHqLoop_')[-1]=='JET_PunchThrough_MC16__1down_Loose':
                    tree_name = 'tHqLoop_JET_PunchThrough_AFII__1down_Loose' 
                elif  tree_t.split('tHqLoop_')[-1]=='JET_PunchThrough_MC16__1up_Loose':
                    tree_name = 'tHqLoop_JET_PunchThrough_AFII__1up_Loose' 
                elif  tree_t.split('tHqLoop_')[-1]=='JET_JER_DataVsMC_MC16__1down_Loose':
                    tree_name = 'tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose' 
                elif  tree_t.split('tHqLoop_')[-1]=='JET_JER_DataVsMC_MC16__1down_PseudoData_Loose':
                    tree_name = 'tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose' 
                elif  tree_t.split('tHqLoop_')[-1]=='JET_JER_DataVsMC_MC16__1up_Loose':
                    tree_name = 'tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose' 
                elif  tree_t.split('tHqLoop_')[-1]=='JET_JER_DataVsMC_MC16__1up_PseudoData_Loose':
                    tree_name = 'tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose' 
                else:
                    tree_name = ''     
            else:
                tree_name = tree_t
       #print(file)
        #if file not in ['mc16a.700205.Sh_2210_ttW_EWK.FS.root','mc16d.700205.Sh_2210_ttW_EWK.FS.root','mc16e.700205.Sh_2210_ttW_EWK.FS.root','mc16a.700168.Sh_2210_ttW.FS.root','mc16d.700168.Sh_2210_ttW.FS.root','mc16e.700168.Sh_2210_ttW.FS.root']: continue
        #if file not in ['ttZ.root']: continue
        #if file not in ['tHq_ytm1']: continue
        #if file not in ['tWH_ytm1']: continue
        print branches_t
        if thread:
            pool.apply_async(UpdateTree,args=(file_path,file,tree_name,branches_t,input_model.BDT_var,input_model.ModelStored,output,branch_name,))
        else:
            UpdateTree(File=file_path,filename = file, tree=tree_name,branches=branches_t, BDT_var = input_model.BDT_var, model =input_model.ModelStored, out_dir =output,branch_out = branch_name, verbose = False)
    
    if thread:
    # Waiting for the processes to end
        pool.close()

    # print "Waiting all asynchronous processes to end"
        pool.join()

################################################################################################
class Model:
    def __init__ (self,t_method,data):
        self.model = eval(method[t_method])
        self.method = t_method
        self.data = data
        self.CV = 'none'
        self.signal=''
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.param = {}
        self.weigth_train = []
        self.weigth_test = []
        self.Alldata = data.copy()
        self.ModelStored =[]
        self.MultiClass = []
        list_of_variable = data.copy().columns.values.tolist()
        #List of variable that they are not included in the mva process
        not_in_bdt_train = ['DSID', 'NEvent','weight_nominal','S/B','eventNumber', 'weight_nominalWtau']
        for var in not_in_bdt_train:
            try:
                list_of_variable.remove(var)
            except:
                #print('WARNING: the variable %s is not included in data'%var)
                pass
        self.BDT_var = list_of_variable
    def ModuleSplit (self, module = 5):     
        try:
            self.X_train,self.X_test,self.y_train,self.y_test,self.weigth_train,self.weigth_test = ModuleSplit(module,self.data,'eventNumber',self.BDT_var,weigth=True)
            self.CV = 'ModuleSplit'
        except:
            print('----------ERROR: ModuleSplit----------')
            print('eventNumber ans S/B fields are needed and data should be pandas dataframe, it is %s'%type(self.data))
    
    def ModuleKfold (self, kfold = 5):   
        try:
            self.X_train,self.X_test,self.y_train,self.y_test,self.weigth_train,self.weigth_test = ModuleKfold(kfold,self.data,'eventNumber',self.BDT_var,weigth=True)
            self.CV = 'ModuleKfold'
        except:
            print('----------ERROR: ModuleKfold----------')
            print('eventNumber and S/B fields are needed and data should be pandas dataframe, it is %s'%type(self.data))

    def SetParams(self, param):
        self.model.set_params(**param)
        self.param = param 
    def GetParams(self):
        print(self.model)
    def GetVariables(self):
        return self.data.columns
    def RestartModel(self):
        del self.model
        self.model = eval(method[self.method])
        self.model.set_params(**self.param)
    def RestartData(self):
        self.data = self.Alldata.copy()
    def AppendModel(self):
        self.ModelStored.append(self.model)
    def MultiClassTarget(self,multi_class):
        df = pd.DataFrame(columns=['DSID','MultiClassTarget'])
        for key in DSID_LIST:
            MultiClass_target = len(multi_class)
            for i , value in enumerate(multi_class):
                if key == value:
                    MultiClass_target=i
            for DSID in DSID_LIST[key]:
                df.loc[len(df)]=[DSID,MultiClass_target]
        df['DSID']=df['DSID'].astype('float64')
        result = pd.merge(self.data,df,how = 'left', on = 'DSID')
        self.data.index = pd.RangeIndex(len(self.data))
        result.index = self.data.index
        self.data['S/B']=result['MultiClassTarget']
        multi_class.append('OtherOA')
        self.MultiClass = multi_class
