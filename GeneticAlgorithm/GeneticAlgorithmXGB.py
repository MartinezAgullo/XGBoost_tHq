#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os,sys
from ROOT import TFile, TTree, TH1D
import ROOT
import numpy as np
import random

import threading
import multiprocessing as mp
import pandas as pd

import matplotlib
matplotlib.use('Agg') # Bypass the need to install Tkinter GUI framework
import matplotlib.pyplot as plt

from StatsGenetic import *


# ATLAS recommend Zn calculator
cs = CommonStats()
# Utility class for performing mutation/cross-over
cross_mutate = cross_mutate()
# Define population size
sys.path.append('../../tHqUtils/')
from  Reader import Root2DfInfo, OpenYaml
sys.path.append('../Utils/')
from ToolsMVA import Model, EvalModel
#sys.path.append('../XGBoost/')
#from XGBoostTool import Fitter, Metrics
pop_size = 30

def rootDefaults() :

    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    
def main():
   
    from optparse import OptionParser
    parser = OptionParser(usage = "usage: %prog arguments", version="%prog")
    parser.add_option("-i","--idir",      dest="idir",                           help="configuration file (default: %default)")
    parser.add_option("-b","--bkg",       dest="bkg",                            help="configuration file (default: %default)")
    parser.add_option("-c","--config",    dest="config",                         help="Channel [OS, SS] (default: %default)")
    parser.add_option("-C","--channel",   dest="channel",                        help="configuration file (default: %default)")
    parser.add_option("-p","--parameter", dest="configparam")
    parser.add_option("-n","--negWeight", dest="negWeight",                      help="Strategy for negative weights [DropNeg/Absolute]")
    parser.add_option("--iteration", dest="iter")
    parser.add_option("-s", "--signal",dest= "signal", help = "Sample to be signal in the classifier (default: %default)")
    parser.set_defaults(config="config/mc16_13TeV_config.ini"
                        ,idir="/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/nominal_Loose/"
                        ,channel="OS"
                        ,bkg=""
                        ,configparam ='ShortMVAtH'
                        ,iter =20
                        ,signal = "tH"
                        ,negWeight="Absolute")
    (options,args) = parser.parse_args()
    
    rootDefaults()

    #Read config file
    if options.channel == "OS" or  options.channel == "SS":
        cfg = OpenYaml("/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_%s.yaml"%options.channel)
    
    if options.channel != "OS" and options.channel != "SS":
        print("Not using dileptau")
        cfg=OpenYaml(options.config)
    
        print("Running GA for the "+str(options.signal)+" process in the "+str(options.channel) +" channel")
    else:
        print("Running GA for the "+str(options.signal)+" process in the '2L "+str(options.channel) +" + Tau'  channel")
    print("Reading : "+str(options.idir))

    global option_g
    #option_g = cfg['Options']
    
    if options.signal == 'tHq':
        options.configparam = 'ShortMVAtHq'
    elif options.signal == 'ttbar':
        options.configparam =  'ShortMVAttbar'
    elif options.signal == 'ttW':
        options.configparam =  'ShortMVAttW'
    elif options.signal == 'Diboson':
        options.configparam =  'ShortMVAVV'
    elif options.signal == 'tZq':
        options.configparam =  'ShortMVAtZq'
    elif options.signal == 'tH':
        options.configparam =  'ShortMVAtH'
    else:
        options.configparam =  'GeneralMVA'

    CfgParameter = cfg[options.configparam]  # Set of variables
    CfgParameter_tune = cfg['XGBoostTuning'] # Hyperparameter configuration
    print("Parameter tuning: " + str(CfgParameter_tune))

    index = 0
    dbFiles = {}

    # Variables from the analysis that are going to be used 
    name_t=[]
    print("Variables used:")
    for var in CfgParameter.values():
        print("  - "+str(var['Name']))
        name_t.append(str(var['Name']))

    
    name_t.append("weight_nominal")
    #name_t.append("m_nbjets")
    #name_t.append("m_njets")
    name_t.append("eventNumber")
    name_t.append('ECIDS_lep1')
    name_t.append('ECIDS_lep2')
    name_t.append('ECIDS_lep3')
    name_t.append('ele_ambiguity_lep1')
    name_t.append('ele_ambiguity_lep2')
    name_t.append('ele_ambiguity_lep3')
    name_t.append('ele_AddAmbiguity_lep1')
    name_t.append('ele_AddAmbiguity_lep2')
    name_t.append('ele_AddAmbiguity_lep3')
    name_t.append('SS_LepHad')

    if options.channel == "OS" or  options.channel == "SS":  name_t.append("weight_nominalWtau")
    branches_t = name_t

    tree_t = 'tHqLoop_nominal_Loose'

    #Read root file from a directory, the path starts form the current directory 
    # An additional "column" is added to the array to store the training data                                                                            

    #array_input= Root2DfInfo('/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/out_V33_3L/', TreeName = tree_t ,branches = branches_t)
    #dbFiles= Root2DfInfo('/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/V34_AtLeast1Bjet_AtLeast1jet_2Tight_2LSS/nominal_Loose/', TreeName = tree_t ,branches = branches_t, signal = options.signal)
    if options.channel == 'SS':
        dbFiles= Root2DfInfo('/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024/nominal_SS/', TreeName = tree_t ,branches = branches_t, signal = options.signal)
        dbFiles.query("SS_LepHad == 1",inplace=True)
    if options.channel == 'OS':
        dbFiles= Root2DfInfo('/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024/nominal_OS/', TreeName = tree_t ,branches = branches_t, signal = options.signal)
        dbFiles.query("SS_LepHad == 0",inplace=True)
    #dbFiles = pd.DataFrame(array_input, columns= columns_t)
    #dbFiles.query("2>=m_nbjets>=1 and  3 >= m_njets >= 1",inplace=True)
    #dbFiles.drop(['m_nbjets','m_njets'],axis=1,inplace=True)
    if options.channel == '3L':
        dbFiles= Root2DfInfo(options.idir, TreeName = tree_t ,branches = branches_t, signal = options.signal)


    # Add tau scale factors. Substitue weight_nominal with weight_nominalWtau
    if options.channel == "OS" or  options.channel == "SS":
        dbFiles['weight_nominal'] = dbFiles['weight_nominalWtau'] 
        dbFiles.drop('weight_nominalWtau', axis=1, inplace=True)


    # Apply PR
    dbFiles.query('ECIDS_lep1==1 & ECIDS_lep2==1 & ECIDS_lep3==1 & ele_ambiguity_lep1<=0 & ele_ambiguity_lep2<=0 & ele_ambiguity_lep3<=0 & ele_AddAmbiguity_lep1<=0 & ele_AddAmbiguity_lep2<=0 & ele_AddAmbiguity_lep3<=0', inplace=True)
    dbFiles.drop(['ECIDS_lep1','ECIDS_lep2','ECIDS_lep3', 'ele_ambiguity_lep1', 'ele_ambiguity_lep2', 'ele_ambiguity_lep3', 'ele_AddAmbiguity_lep1', 'ele_AddAmbiguity_lep2', 'ele_AddAmbiguity_lep3','SS_LepHad'], axis=1, inplace=True)

    
    m_model = Model('XGBoost',dbFiles)
    # The m_model.Alldata keps the original weights. These can be retrieved with m_model.RestartData()
    #if options.channel == "OS":
    if options.negWeight not in ["DropNeg", "Absolute"]:
        print("Error with options.negWeight")
        exit()
    if options.negWeight == "DropNeg":
        print("[GA] Using posively weighted events only")
        m_model.data.query('200>weight_nominal > 0',inplace=True)
    #if options.channel == "SS":
    if options.negWeight == "Absolute":
        print("[GA] Using aboslute weights")
        m_model.data['weight_nominal']=m_model.data['weight_nominal'].abs()

    m_model.data['weight_nominal']=m_model.data['weight_nominal']*140.5
    m_model.ModuleSplit()
    param_tHq = {'max_depth': 3
             #,'objective': 'multi:softmax'
             ,'objective': 'binary:logistic'
             ,'learning_rate':0.025
             ,'n_estimators':1500                                                                                     
             ,'min_child_weight' :0.1 
             ,'tree_method':'gpu_hist'
             #,'max_delta_step': 2
             #,'min_split_loss':0                                                                                
             #,'eval_metric': ['error',"auc","logloss"]
             ,'n_jobs':-1
             ,'scale_pos_weight': 200 #(m_model.Alldata[m_model.Alldata['S/B']==0]['weight_nominal'].sum()/m_model.Alldata[m_model.Alldata['S/B']==1]['weight_nominal'].sum())/4#(bkg_yield/sig_yield)/2
            }
    
    m_model.SetParams(param_tHq)
    #Initial values
    #Scan for Background
    #kg_yield, bkg_error, bkg_raw =  get_yield(dbFiles[dbFiles['S/B']==0])
    #ig_yield, sig_error, sig_raw =  get_yield(dbFiles[dbFiles['S/B']==1])
    #ig = (sig_yield/np.sqrt(bkg_yield + sig_yield))
    #ig_error = sig*np.sqrt(((1-sig_yield*(sig_yield+bkg_yield))/sig_yield)**2*sig_error**2+sig_error**2/(2*(sig_yield+bkg_yield))**2)
    #n = sig_yield/(bkg_yield+sig_yield)
    #rint('---------Initial values----------')
    #rint('Signal(%i):       %f +- %f'%(sig_raw,sig_yield,sig_error))
    #rint('Background(%i):    %f +- %f'%(bkg_raw,bkg_yield, bkg_error))
    #rint('Significance: %f +- %f'%(sig,sig_error))
    #rint('---------------------------------')
    #Description table for signal and background
    #able_description(dbFiles[dbFiles['S/B']==0],name='bkg')
    #able_description(dbFiles[dbFiles['S/B']==1],name='signal')

    # Generate initial population
    pop_size = 30
    init_df = init_population(pop_size ,CfgParameter_tune)  # pop_ize = 30 :: There 30 different configurations of hyperparameters
    Zn_best = 0.
    output_df = pd.DataFrame(columns= init_df.columns)
    print("[GA] init_df has a population of hypeparameters to be evaluated and mutated. ")
    print("[GA] Initial population for GA: init_df")
    print("[GA] Adding by hand a few rows into init_df")
    #      scale_pos_weight  min_child_weight  learning_rate       Auc  LogLoss    Zn
    init_df.loc[int(pop_size + 1)] = [268.838, 0.50, 0.1237, 0.000001, 0.001, -99.0] # Add current best BDT(tHq OS) 
    init_df.loc[pop_size + 2] = [1.1, 0.05, 0.025, 0.000001, 0.001, -99.0]      # Add current best BDT(ttbar)
    init_df.loc[pop_size + 3] = [83.21, 0.026, 0.04, 0.000001, 0.001, -99.0]    # Add current best BDT(tHq SS)
    init_df.loc[pop_size + 4] = [179.06, 0.059, 0.091, 0.000001, 0.001, -99.0]  # Add previous GA-optimal BDT(tHq OS)
    init_df.loc[pop_size + 5] = [132.739, 0.0126, 1.5634, 0.000001, 0.001, -99.0]  # Add previous GA-optimal BDT(tHq OS)
    init_df.loc[pop_size + 6] = [38.4, 0.0161, 1.49, 0.000001, 0.001, -99.0]  # Add previous GA-optimal BDT(tH OS)
    init_df.loc[pop_size + 7] = [83.8, 0.0139, 0.42, 0.000001, 0.001, -99.0]  # Add previous GA-optimal BDT(tH OS) ]
    init_df.loc[pop_size + 8] = [0.43, 0.041, 0.023, 0.000001, 0.001, -99.0]      # Add current best BDT(ttbar)
    init_df.loc[pop_size + 9] = [60.0, 0.056, 0.0334, 0.000001, 0.001, -99.0]      # Add current best BDT(ttbar) 
    init_df.loc[pop_size + 10] = [300, 1, 0.5, 0.000001, 0.001, -99.0]  # Add tH OS good
    init_df.loc[pop_size + 11] = [28, 0.012, 0.22, 0.000001, 0.001, -99.0]  # Add previous GA-optimal BDT(tH OS) ]  
    init_df.loc[pop_size + 12] = [300, 0.41, 0.5, 0.000001, 0.001, -99.0] 
    init_df.loc[pop_size + 13] = [99.94, 0.0158, 0.142043, 0.000001, 0.001, -99.0]  # tH OS
    init_df.loc[pop_size + 14] = [0.34, 0.01, 0.251, 0.000001, 0.001, -99.0] # ttbar OS
    init_df.loc[pop_size + 15] = [63.13, 0.014, 0.5, 0.000001, 0.001, -99.0] # test for SS tH
    init_df.loc[pop_size + 16] = [30, 0.012, 0.25, 0.000001, 0.001, -99.0] # test for SS tH
    init_df.loc[pop_size + 17] = [45.2, 0.131, 1.46, 0.000001, 0.001, -99.0] # test for SS tH 
    init_df.loc[pop_size + 18] = [63.7, 0.0157, 1.5, 0.000001, 0.001, -99.0] # test OS tH
    pop_size = pop_size + 18
    print(init_df)
    NStep = 0
    print('[GA] Number of total iter: %i'%int(options.iter))

    
    print("[GA] **********************")
    print("[GA] **      Start GA    **")
    print("[GA] **********************")
    while NStep<int(options.iter):
        print("[GA] Iteration "+str(NStep)+"/"+str(options.iter))
        print("[GA] Evalutae fitness fuctions for each set of hyperparameteres:")
        # Fitness function
        evaluate(init_df,m_model) # Evalues the model with the hyperparams in init_df.  Writes AUC, logloss and Zn into init_df
        init_df = init_df.sort_values('Zn', ascending=False)
        #init_df.sort_values(by=['Zn'],inplace=True) 
        init_df.reset_index(drop=True,inplace=True)
        print("[GA] Evaluation of metrics completed for the collection of hyperparams in the iteration %i"%int(NStep))
        print("[GA] The inint_df sorted by Zn:")
        print(init_df) #init_df with the metrics and reordered according to the Zn score

        # Select the half of the population that perfoms worst 
        #Zn_condition = init_df.iloc[int(pop_size/2.)-1]['Zn'] # Central Zn value
        #Zn_best = init_df.iloc[pop_size - 1]['Zn']            # Zn 
        #print("[GA] Zn_condition = %s"%Zn_condition)
        #print("[GA] Zn_best = %s"%Zn_best)
        #print("[GA] Hypeparams with Zn lower than Zn_condition are removed from init_df")
        rows_to_delete = len(init_df) // 2

        # Print current best
        print("\n [GA] -> Best set of Hyperameters for iter %i"%NStep)
        print(init_df.iloc[0]) # First row of init_df
        output_df.loc[NStep] = init_df.iloc[0]
        N_StuckIters = 3
        std_StuckThereshold = 0.001
        if NStep>= N_StuckIters and output_df.tail(N_StuckIters).std()['Zn']< std_StuckThereshold :
            # The standard deviation of the Zn of the last three itartions is below "std_StuckThereshold"
            print("[GA] Stoping iterations at "+str(NStep)+" step because the Zn has not improved in the last "+str(N_StuckIters) +" iterations.")
            break
        # Drop worst performing half
        #init_df.drop(init_df.index[:int(pop_size/2)],inplace=True)
        init_df = init_df.iloc[:rows_to_delete] #delete latest rows
        init_df.reset_index(drop=True,inplace=True)
        print("[GA] Drop the worst performing half of init_df")
        print(init_df) #init_df

        # Do crossover/mutation
        print("[GA] Do crossover and mutation")
        df_cm = cross_mutate.run(init_df)
        init_df = init_df.append(df_cm,ignore_index=True)
        #print 'TEST_TO_APPEND'
        #print df_cm
        init_df.reset_index(drop=True,inplace=True)
        init_df.drop_duplicates(inplace=True)
        #print 'TEST_2'
        #print init_df
        # Add new values?
        missing = int(pop_size - init_df.shape[0])
        if missing == 0:
            print('[GA] Moving on')
        else:
            init_df = init_df.append(init_population(missing,CfgParameter_tune))
            init_df = init_df.reset_index(drop=True)
        #print 'TEST_3'
        #print init_df
        print("\n [GA] -> Added {} values".format(missing))

        print("\n [GA] -> Saved Results")
        print( init_df.query('Zn>0') )
        #init_df = init_df.append(mutate(init_df))
        NStep = NStep + 1
    ##print(output_df)


    #output_df.plot(subplots=True)

    #plt.tight_layout()
    #plt.savefig('output.png')
    #plt.close()
    
    # Store the result con csv file #
    if options.negWeight == "Absolute":
        output_df.to_csv('output_GA_tunning_%s_%s_AbsoluteWeight.csv'%(options.channel,options.signal),index=False)
    if options.negWeight == "DropNeg":
        output_df.to_csv('output_GA_tunning_%s_%s_PositiveOnlyWeight.csv'%(options.channel,options.signal),index=False)
    sigcolumns =['Auc','LogLoss','Zn']
    try:
        EvolutionPlot(output_df.drop(sigcolumns,axis=1),2,3)
        plt.tight_layout()
        plt.savefig('evol_cut_%s_%s.png'%(options.channel,options.signal))
        plt.close()
        EvolutionPlot(output_df[sigcolumns],2,2)
        plt.tight_layout()
        plt.savefig('evol_GA_%s_%s.png'%(options.channel,options.signal))
        plt.close()
    except:
        print('WARNING:Resume plots not produced')

def EvolutionPlot(df,size_x,size_y):
    col = df.columns
    fig, axes = plt.subplots(nrows=size_x, ncols=size_y)
    n = 0
    color = ['b','g','c','m','y','r','orange','purple','gray']
    if len(color)<len(col):
        while len(color)<len(col):
            color.append(color)
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
 	        try:
	            df[col[n]].plot(legend = True,ax=axes[i,j],color = color[n])
	            n=n+1
	        except:
	            print("[GA] Last column tried :: "+str(n))

def table_description(df,name='All'):
        
    df_desc= df.describe(percentiles=[.1,.25, .5, .75,.9])
    
    #create a subplot without frame
    plot = plt.subplot(111, frame_on=False)

    #remove axis
    plot.xaxis.set_visible(False) 
    plot.yaxis.set_visible(False) 

    #create the table plot and position it in the upper left corner
    table = pd.plotting.table(plot, df_desc,loc='upper right')
    #table.auto_set_font_size(False)
    #table.set_fontsize(14)
    #save the plot as a png file
    plt.title('Table description for %s'%name)
    plt.savefig('desc_%s.pdf'%name)
    plt.close()

#@jit(nopython=True, parallel=True)
def rand_value(low, high,dp=0):

    if dp == 5 :
        return random.randrange(low,high,dp)

    return round(np.random.uniform(low, high),dp)

#@jit

def init_population(pop_size,parameters):

    columns = []
    sigcolumns =['Auc','LogLoss','Zn']
    
    for var in parameters.values():
        columns.append(str(var['Name']))
 

    dfcolumns = np.array(columns)
    dfcolumns = np.append(dfcolumns,sigcolumns)
    
    df = pd.DataFrame(columns=dfcolumns)

    dfvars = {}
    #print 'TEST'
    #print parameters.values()
    for i in range(pop_size):
        row = []

        for var in parameters.values():
            if 'Type' in var:
                if var['Type']=='Log':
                    l_min = np.log10(float(var['Min']))
                    l_max = np.log10(float(var['Max']))
                    try: l_random = rand_value(l_min,l_max,int(var['Seed']))
                    except:
                        print("[GA] Error initialising hyperparameter "+str(var))
                        exit()
                    row.append(10**l_random)
                else:
                    try:
                        row.append(rand_value(float(var['Min']),float(var['Max']),int(var['Seed'])))
                    except:
                        print("[GA] Error initialising hyperparameter "+str(var))
                        exit()
            else:
                row.append(rand_value(float(var['Min']),float(var['Max']),int(var['Seed'])))

        row.append(0.000001)
        row.append(0.001)        
        row.append(-99)
        
        row = np.asarray(row)
        df.loc[i] = row

    return df    

# Evaluate Zn for given cuts
def StringCut(parameters):

    extra = ''
    for var in parameters.values():
        if 'Extra' in var:
            extra = var['Extra']
        if 'cut_aux_1' in locals():
            cut_aux_1 = cut_aux_1 + ' and '+extra+'('+var['Name']+')'+option_g[var['Option']] +'{}'
        else:
            cut_aux_1 = extra+'('+var['Name']+')'+option_g[var['Option']]+'{} '

        if 'cut_aux_2' in locals():
            cut_aux_2 = cut_aux_2 + ',row[\''+var['Name']+'\']'
        else:
            cut_aux_2 = 'row[\''+var['Name']+'\']'
    aux_cut = '(\''+cut_aux_1+'\').format('+cut_aux_2+')'
    #print(aux_cut)
    return aux_cut


# evaluate: Evaluates m_model for each hyperparam in init_df.
def evaluate(df,m_model):
    benchmark = 0
    #aux_cut = StringCut(parameters)
    #print(parameters)
    print("[GA evaluate] Evalutaing diferent models in init_df")
    for index, row in df.iterrows(): #Iterate over df rows 
        # row[:-3] are all the values of the row except the last three columns
        # row[:-3].to_dict() converts the row[:-3] output into dictionary {column1:value, coumn2:value, ....}
        print("[GA evaluate]  Evaluating model with hyperparams:")
        print(row[:-3])

        if row['scale_pos_weight'] <= 0 or row['min_child_weight'] <= 0 or row['learning_rate'] <= 0:
            if row['scale_pos_weight'] <= 0:
                print("Negative scale_pos_weight not allowed. Changing it to 0.0001")
                row['scale_pos_weight'] = 0.00001
            if row['min_child_weight'] <= 0:
                print("Negative min_child_weight not allowed. Changing it to 0.0001")
                row['min_child_weight'] = 0.00001
            if row['learning_rate'] <= 0:
                print("Negative learning_rate not allowed. Changing it to 0.0001")
                row['learning_rate'] = 0.00001


        metrics = EvalModel(m_model,row[:-3].to_dict()) 
        print(row[:-3].to_dict()) 
        #exec(aux_cut)
        #cuts = eval(aux_cut)
        #sig_yield, sig, sn, Zn1 = scan(cuts,dbFiles)
        #row['SigYield'] = sig_yield
        row['Auc'] = metrics[0]      # Add column with AUC
        row['LogLoss'] = metrics[1]  # Add column with logloss
        if metrics[0] == 0.5:        # If AUC = 0.5 the Zn is -99
            Zn1 = -99
        else:
            if metrics[1]<1:
                Zn1= 1./(1.-metrics[0]) - np.log(metrics[1])
            else: # if logloss > 1 
                Zn1= 1./(1.-metrics[0]) - 1.0/metrics[1]
                Zn1= -99
                # pablo: I would use something like Zn = -99 here

            if math.isnan(Zn1):
                Zn1=-99
        row['Zn'] = Zn1             # Add row with Zn
        print("[GA evaluate]  Result of training :")
        print(row.to_dict())
        print(" ")
        print(" ")
        #row['SigYield'] = sig_yield[sigBenchmark]
        #row['Sig'] = sig[sigBenchmark]
        #row['SN'] = sn[sigBenchmark]
        #row['Zn'] = Zn1[sigBenchmark]
        #for x in range(1,len(signalDSIDs)) :
        #    row['SigYield'+"_"+str(signalDSIDs[x])] = sig[str(signalDSIDs[x])]
        #    row['Sig'+"_"+str(signalDSIDs[x])] = sig[str(signalDSIDs[x])]
        #    row['SN'+"_"+str(signalDSIDs[x])] = sn[str(signalDSIDs[x])]
        #    row['Zn'+"_"+str(signalDSIDs[x])] = Zn1[str(signalDSIDs[x])]


# Calculates yields and errors for all samples
def scan(cuts,dbFiles):

    total_bkg_yield = 0.
    total_bkg_error = 0.
    total_sig_yield = {}
    total_sig_error = {}

    #Scan for Background
    bkg_yield, bkg_error, bkg_raw =  get_yield(dbFiles[dbFiles['S/B']==0],cuts)
    total_bkg_yield = bkg_yield
    total_bkg_error = bkg_error

    #Scan for Signal
    sig_yield, sig_error, sig_raw =  get_yield(dbFiles[dbFiles['S/B']==1],cuts)
    total_sig_yield = sig_yield
    total_sig_error = sig_error

    #print total_sig_yield, total_bkg_yield+total_sig_yield['W+jets'],total_sig_yield['W+jets']/(total_bkg_yield+total_sig_yield['W+jets'])

#    sig = {}
#    sn =  {}
#    cs_sig = {}

    sig = 0                                                                                                                                                                                              
    sn =  0                                                                                                                                                                                              
    cs_sig = 0

#    for sam in total_sig_yield :
    if total_sig_yield != 0 and total_bkg_yield != 0:# and total_sig_yield > 2.:
        sig = (total_sig_yield/np.sqrt(total_bkg_yield + total_sig_yield))
        sig_error = sig*np.sqrt(((1-total_sig_yield*(total_sig_yield+total_bkg_yield))/total_sig_yield)**2*total_sig_error**2+total_sig_error**2/(2*(total_sig_yield+total_bkg_yield))**2)
        sn = total_sig_yield/(total_bkg_yield+total_sig_yield)
        cs_sig = cs.AsymptoticPoissonPoissonModel(total_sig_yield, total_bkg_yield, total_bkg_error)
        #cs_sig = cs.AsymptoticPoissonPoissonModel(total_sig_yield, total_bkg_yield, total_bkg_error)
    else : 
        sig = 0.001
        sn = -0.01
        cs_sig = -99
        
    return  total_sig_yield, sig, sn, cs_sig

# Calculate yield for single sample with given cuts

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
'''
    try :
        tree = file[0].Get('ResultsDataTree')
        hist = TH1D("hist","hist",1,0,1000)
        hist.Sumw2()
        # Weights string!
        weight = cuts + ' * nLumi * combinedWeight'
        #print weight
        raw = tree.GetEntries(cuts)
        tree.Draw('1>>hist', weight)
        return hist.GetBinContent(1), hist.GetBinError(1), raw
    except :
        return 0,0,0

'''
if __name__=='__main__':
    main()    

 
