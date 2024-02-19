
import sys, os
import multiprocessing
import pandas as pd
from optparse import OptionParser

sys.path.append('./XGBoost/')
from XGBoostTool import Fitter, FeatureImportance,OptimizeFeature
sys.path.append('../tHqUtils/')
from Reader import Root2DfInfo, OpenYaml

sys.path.append('./Utils/')
from ToolsMVA import Model, ApplyPrediction, GradientOptimize, RandomOptimize,LoadModel, ConfussionMatrix, EvalModel

parser = OptionParser(usage = "usage: %prog arguments", version="%prog")
parser.add_option("-m", "--model", dest="model")
parser.add_option("-t", "--tree-name", dest="treeName",     help="set the tree name to process (default: %default)")
parser.add_option("-i", "--inputpudir",  dest="inputdir",    help="input directory (default: %default)")
parser.add_option("-c", "--configfile", dest ='config_file', help = "configuration file with variables (default: %default)")
parser.add_option("-s", "--signal",dest= "signal", help = "Sample to be signal in the classifier (default: %default)")
parser.add_option("-o", "--output",dest= "output", help = "Folder for the output BDT (default: %default)")
parser.add_option("--mode", dest= "mode", help = "Execution mode of the program: Train, Load, Optimize (default: %default)")
parser.add_option("--channel", dest= "channel", help = "Choose between possible channels of the dilpetau analysis: SS or OS (default: %default)")
parser.add_option("--name", dest= "name", help = "Add a string to the current name of the outputs (default: %default)")
parser.set_defaults(model='XGBoost', treeName = 'tHqLoop_nominal_Loose'
                    , config_file='/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml'
                    , inputdir = '/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024/nominal_OS/'
                    , signal='tH'
                    , output='output_BDT'
                    , mode = 'Train'
                    , channel= 'SS'
                    , name='')
(options,args) = parser.parse_args()

m_name = options.channel if options.name=='' else options.channel+'_'+options.name



print("BDT("+str(options.signal)+" - "+str(options.channel)+")")


def Branches(GroupVar=' '):
    #Read config file

    cfg = OpenYaml(options.config_file)
    print("Config: "+str(options.config_file))
    print("Group var: "+str(GroupVar))
    # Variables from the analysis that are going to be used 
    CfgParameter = cfg[GroupVar]

    cfg_dict = {}
    for var in CfgParameter.values():
        if 'Label_name' in var.keys():
            cfg_dict[str(var['Name'])]=str(var['Label_name'])
        else:
            cfg_dict[str(var['Name'])]=str(var['Name'])

    return cfg_dict

def Read(GroupVar=' ', tree_name = options.treeName, branches=''):
    #Read root file from a directory, the path starts form the current XGBoost directory 
    # An additional "column" S/B , DSID, NEvent (number of events in each DSID) is added to the array to store the training data  where 0 means bkg and 1 means signal  
    df_input = Root2DfInfo(options.inputdir, TreeName = tree_name, branches=branches, signal = options.signal)
    return df_input


if options.mode == 'Optimise': options.mode = 'Optimize'

#m_model = Model(options.model,Read())
#df = pd.DataFrame(columns= Branches())

if options.signal == 'tH':
    branch = 'ShortMVAtH'
    #branch = 'GeneralMVA'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
elif options.signal == 'tHq':
    #branch = 'ShortMVAtHq'
    branch = 'baseline_tHq'
    #branch = 'GeneralMVA'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
elif options.signal == 'ttbar':
    branch = 'ShortMVAttbar'
    #branch = 'GeneralMVA'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
elif options.signal == 'tWH':
    branch = 'GeneralMVA'
    if options.channel == 'OS':  branch = 'ShortMVAtWH'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
elif options.signal == 'ttW':
    branch = 'ShortMVAttW'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
elif options.signal == 'Diboson':
    branch = 'ShortMVAVV'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
elif options.signal == 'Zjet':
    branch = 'ShortMVAZjets'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
elif options.signal == 'tZq':
    branch = 'ShortMVAtZq'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
elif options.signal == 'ttX':
    branch = 'ShortMVAttX'
    if options.mode == 'Optimize': branch = 'GeneralMVA'
else:
    branch = 'GeneralMVA'


tree_name = options.treeName
if tree_name == 'tHqLoop_alternative_sample': tree_name = 'tHqLoop_nominal_Loose'
print tree_name

dict_branch = Branches(branch)
branches =[]
branches+=dict_branch.keys()
branches.append('weight_nominal')
if options.channel == "OS" or  options.channel == "SS": 
    branches.append('weight_nominalWtau')
    branches.append('OS_LepHad')
branches.append('eventNumber')


if options.mode =='Load':
    df = pd.DataFrame(columns=branches)
else:
    df = Read(branch,tree_name,branches)
#df = pd.DataFrame(columns= Branches(branch))

if options.channel == "OS" or  options.channel == "SS":
    if df['weight_nominal'].equals(df['weight_nominalWtau']):
        print("Warning: The weights with the tau fake factors are not included")
    else: print("Weights with the tau fake factors included")


lumi = 140.5
if options.mode=='Train':
    print("Raw events before training (with weight_nominal)")
    print("    Total ::    \t\t"+str(len(df)))
    print("    Of "+str(options.signal)+" "+str(options.channel)+" process  ::\t" + str((len(df[df['S/B']==1]))))
    print("    Others ::   \t\t" + str((len(df[df['S/B']==0]))))
    yields_total = (df['weight_nominal'].sum())*lumi
    yields_target = (df[df['S/B']==1]['weight_nominal'].sum())*lumi
    print("Total events with weight_nominal :: " + str(yields_total))
    print(str(options.signal)+" "+str(options.channel)+ " events with weight_nominal :: " + str(yields_target))
    if options.channel == "OS":
        print("OS ::  Removing SS events")
        df = df.query('OS_LepHad==1')
    if options.channel == "SS":
        print("SS ::  Removing OS events")
        df = df.query('OS_LepHad==0')

#print options.signal
#if branch == 'GeneralMVA': exit()


# Apply tau scale factors
if options.channel == "OS" or  options.channel == "SS":
    df['weight_nominal'] = df['weight_nominalWtau'] # The column weightNominal has the weights with the fake factors. 
    if df['weight_nominal'].equals(df['weight_nominalWtau']): print("Tau scale factor applied")
    df.drop('weight_nominalWtau', axis=1, inplace=True) #Droping the column weight_nominalWtau. 
    df.drop('OS_LepHad', axis=1, inplace=True)
    #df.weight_nominal.fillna(df.weight_nominalWtau, inplace=True)
    #print(df['weight_nominal'])


if options.mode=='Train':
    yields_total = (df['weight_nominal'].sum())*lumi
    yields_target = (df[df['S/B']==1]['weight_nominal'].sum())*lumi
    print("Total events with weight_nominalWtau :: " + str(yields_total))
    print(str(options.signal)+" "+str(options.channel)+ "events with weight_nominalWtau :: " + str(yields_target))

m_model = Model(options.model,df)
m_model.signal = options.signal



if options.model == 'XGBoost':

    #scale_pos = (m_model.Alldata[m_model.Alldata['S/B']==0]['weight_nominal'].sum()/m_model.Alldata[m_model.Alldata['S/B']==1]['weight_nominal'].sum())
    #print('theo scale positive: %f'%scale_pos)
    #m_model.data.query('m_nNoBjets>=1 and m_nbjets>=1  and abs(LeptonChargeSum)==1 and isTight_lep1==1 and isTight_lep2==1 and isTight_lep3==1 and m_min_diff_mass>10',inplace=True)
    #m_model.Alldata.query('m_nNoBjets>=1 and m_nbjets>=1  and abs(LeptonChargeSum)==1 and isTight_lep1==1 and isTight_lep2==1 and isTight_lep3==1 and m_min_diff_mass>10',inplace=True)
    #m_model.data.query('isTight_lep1==1 and isTight_lep2==1 and isTight_lep3==1',inplace=True)
    #m_model.Alldata.query('isTight_lep1==1 and isTight_lep2==1 and isTight_lep3==1',inplace=True)

    ###############################
    # Hyperparameters for each channnel in XGBoost
    ###############################
    ##OS Anlysis
    if options.channel == 'OS':
        param_tHq = {'max_depth': 4
                 #,'objective': 'multi:softmax'
                 ,'objective': 'binary:logistic'
                 ,'learning_rate': 0.1237
                 ,'n_estimators':1500                                                                                     
                 ,'min_child_weight': 0.50
                 ,'tree_method':'gpu_hist'
                 #,'max_delta_step': 2                                                                                
                 #,'eval_metric': ['error',"auc","logloss"]
                 ,'n_jobs':-1
                 ,'scale_pos_weight': 268.83883206654883
                }
        #Smaller learning rates generally require more trees to be added to the model.
        param_ttbar = {'max_depth': 4
                #,'objective': 'multi:softmax'
                ,'objective': 'binary:logistic'
                ,'learning_rate': 0.251
                ,'n_estimators':1500  
                ,'min_child_weight': 0.01 
                ,'tree_method':'gpu_hist'
                #,'min_split_loss':0 
                #,'max_delta_step': 2                                                                                
                ,'n_jobs':-1
                ,'scale_pos_weight': 0.34
                }
        param_tH = {'max_depth': 4
                 ,'objective': 'binary:logistic'
                 ,'learning_rate': 0.5
                 ,'n_estimators':1500
                 ,'min_child_weight': 0.014
                 ,'tree_method':'gpu_hist'
                 ,'n_jobs':-1
                 ,'scale_pos_weight': 100
                }

        #################################
        #Files with BDT trainned to load
        #################################
        if options.mode =='Load' or options.mode=='OnlyPlot':
            #OS
            load_tHq=['/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K0_OS__GoodToGo.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K1_OS__GoodToGo.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K2_OS__GoodToGo.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K3_OS__GoodToGo.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K4_OS__GoodToGo.json']

            load_ttbar_newNTuples=['/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K0_OS_FinalModel.json',
                        '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K1_OS_FinalModel.json',
                        '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K2_OS_FinalModel.json',
                        '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K3_OS_FinalModel.json',
                        '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K4_OS_FinalModel.json']

            load_ttbar=['/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K0_OS_GoodToGo.json',
                        '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K1_OS_GoodToGo.json',
                        '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K2_OS_GoodToGo.json',
                        '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K3_OS_GoodToGo.json',
                        '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_ttbar_K4_OS_GoodToGo.json']


            load_tH=['/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K0_OS__FinalModel.json',
                     '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K1_OS__FinalModel.json',
                     '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K2_OS__FinalModel.json',
                     '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K3_OS__FinalModel.json',
                     '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K4_OS__FinalModel.json']
                
    ###############################
    ##SS Anlysis
    elif options.channel == 'SS':
        param_tHq = {'max_depth': 4
                 #,'objective': 'multi:softmax'
                 ,'objective': 'binary:logistic'
                 ,'learning_rate':0.04#0.24
                 ,'n_estimators':1500                                                                                     
                 ,'min_child_weight' :0.026
                 ,'tree_method':'gpu_hist'
                 #,'max_delta_step': 2                                                                                
                 #,'eval_metric': ['error',"auc","logloss"]
                 ,'n_jobs':-1
                 ,'scale_pos_weight': 83.21#136.7
                }
        param_tWH = {'max_depth': 4,
                     'objective': 'binary:logistic'
                     ,'learning_rate':0.04
                     ,'n_estimators':1500
                     ,'min_child_weight' :0.026
                     ,'tree_method':'gpu_hist'
                     ,'n_jobs':-1
                     ,'scale_pos_weight': 83.21
                     }

        param_ttbar = {'max_depth': 4
                       ,'objective': 'binary:logistic'
                       ,'learning_rate': 0.15
                       ,'n_estimators':1500
                       ,'min_child_weight': 0.00012
                       ,'tree_method':'gpu_hist'
                       ,'n_jobs':-1
                       ,'scale_pos_weight': 3
                      }
        param_ttX = {'max_depth': 4,
                     'objective': 'binary:logistic'
                     ,'learning_rate':0.01
                     ,'n_estimators':1500
                     ,'min_child_weight' :0.1
                     ,'tree_method':'gpu_hist'
                     ,'n_jobs':-1
                     ,'scale_pos_weight': 1
                     }
        param_tH = {'max_depth': 4 #SameSign
                 ,'objective': 'binary:logistic'
                 ,'learning_rate': 0.1
                 ,'n_estimators':1500
                 ,'min_child_weight': 0.01
                 ,'tree_method':'gpu_hist'
                 ,'n_jobs':-1
                 ,'scale_pos_weight':38
                }
        #################################
        #Files with BDT trainned to load
        #################################
        if options.mode =='Load' or options.mode=='OnlyPlot':
            #2LSS
            load_tHq_trash=['/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K0_SS_GoodToGo_SameSign.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K1_SS_GoodToGo_SameSign.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K2_SS_GoodToGo_SameSign.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K3_SS_GoodToGo_SameSign.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K4_SS_GoodToGo_SameSign.json']

            load_tHq=['/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K0_SS_good_to_go_ss.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K1_SS_good_to_go_ss.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K2_SS_good_to_go_ss.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K3_SS_good_to_go_ss.json',
                      '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tHq_K4_SS_good_to_go_ss.json']

            load_tH=['/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K0_SS_Pffff_Bueno.json',
                     '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K1_SS_Pffff_Bueno.json',
                     '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K2_SS_Pffff_Bueno.json',
                     '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K3_SS_Pffff_Bueno.json',
                     '/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/model_tH_K4_SS_Pffff_Bueno.json']
            load_ttX=['']
            load_ttbar=['']
            load_ttW=['']
            load_tZq= ['']

    elif options.channel == '3L':
        param_tHq = {'max_depth': 4
                 #,'objective': 'multi:softmax'
                 ,'objective': 'binary:logistic'
                 ,'learning_rate':0.18936512548966736
                 ,'n_estimators':1500                                                                                     
                 ,'min_child_weight' :0.013427649611378635
                 ,'tree_method':'gpu_hist'
                 #,'max_delta_step': 2                                                                                
                 #,'eval_metric': ['error',"auc","logloss"]
                 ,'n_jobs':-1
                 ,'scale_pos_weight': 324.33961734934934
                }  
        param_ttbar  = {'max_depth': 4
                ,'objective': 'binary:logistic'
                 ,'learning_rate': 0.015153048036987023
                 ,'n_estimators':1500
                 ,'min_child_weight': 0.0047315125896148025
                 ,'tree_method':'gpu_hist'
                 ,'n_jobs':-1
                 ,'scale_pos_weight': 3.6140986263961334
                }


    else:
        print('WARNING: Set of hiperparamters not included')

    ###############################
    #Include params to the model
    param = {}
    #param = param_ttW
    if options.signal == 'tHq':
        param = param_tHq
    elif options.signal == 'tWH':
        param = param_tWH
    elif options.signal == 'ttbar':
        param = param_ttbar
    elif options.signal == 'ttW':
        param = param_ttW
        #param = param_ttWSherpa
    elif options.signal == 'ttW_Sherpa':
        param = param_ttWSherpa
    elif options.signal == 'Diboson':
        param = param_VV
    elif options.signal == 'tZq':
        param = param_tZq
    elif options.signal == 'ttX':
        param = param_ttX
    elif options.signal == 'tH':
        param = param_tH
    else:
        print("WARNING: No parameters defined for target process")

    m_model.SetParams(param)

    if options.mode=='Optimize':
        #Splitting trainning / test 80% / 20%
        m_model.ModuleSplit()
        #Optimize list of features
        OptimizeFeature(m_model,name=m_name)

    elif options.mode=='Train':

        # Apply tau scale factors
        #print("Tau scale factors - Total events weighted with weight_nominal \t\t ::"+str(m_model.data['weight_nominal'].sum()) )
        #print("Tau scale factors - Total target process events weighted with weight_nominal \t ::"+str(m_model.data[m_model.data['S/B']==1]['weight_nominal'].sum()) )
        #m_model.data['weight_nominal'] = m_model.data['weight_nominalWtau'] # The column weightNominal has the weights with the fake factors. 
        #m_model.data.drop('weight_nominalWtau', axis=1, inplace=True)
        #print("Tau scale factors - Total events weighted with weight_nominalWtau) \t ::"+str(m_model.data['weight_nominal'].sum()) )
        
        #Ignore negative weight
        #if options.channel == 'OS': 
        if False:
            print("Removing negative weights")
            m_model.data.query('200 > weight_nominal > 0',inplace=True) # Only positively weighted events for training
        #if options.channel == 'SS': 
        if True:
            print("Using abslute weights of the events")
            m_model.data['weight_nominal'] = m_model.data['weight_nominal'].abs() # Absolute weights

        
        print("Raw events in training (after applying neg. weights strategy)")
        print("    Total ::    \t\t"+str(len(df)))
        print("    Of "+str(options.signal)+" process  :: \t" + str((len(m_model.data[m_model.data['S/B']==1]))))
        print("    Others ::   \t\t" + str((len(m_model.data[m_model.data['S/B']==0]))))
        
        #Splitting trainning / test 80% / 20%
        #m_model.ModuleSplit()
        print("[mva_runner]  Kfolding (before train, no negative weights)")
        m_model.ModuleKfold()

        print("[mva_runner]  Model parameters:")
        print(param)

        print("[mva_runner]  Model features:")
        print(dict_branch)

        #Train the model
        print("[mva_runner]  Train the model")
        df = Fitter(m_model,bool_skf=False,save=True, name =m_name)
            
        #Feature importance of the model
        print("[mva_runner]  Draw feature importance")
        FeatureImportance(m_model,graphic=True,name = m_name,label=dict_branch)

        #Restart to initial data
        print("[mva_runner]  Restart data :: Use all weights")
        m_model.RestartData()

        #Splitting trainning test 80% / 20%
        #m_model.ModuleSplit()
        print("[mva_runner]  Kfolding (on all weights)")
        m_model.ModuleKfold()

        #Apply BDT responde to MC
        #Return BDT_response, BDT_cumulative_response, KS test and metrics (auc,logloss and f1-score)
        print("[mva_runner]  Apply BDT response")
        df = ApplyPrediction(m_model, name = m_name)

    
    #elif options.mode=='OnlyPlot':
    #    ###Load the BDT trained to do Plots
    #    if options.signal == 'tHq':
    #        load_file = load_tHq
    #    elif options.signal == 'ttbar':
    #        load_file = load_ttbar
    #    elif options.signal == 'ttW':
    #        load_file = load_ttW
    #    elif options.signal == 'Diboson':
    #        load_file = load_VV
    #    elif options.signal == 'tZq':
    #        load_file = load_tZq
    #    else:
    #        print('ERROR: Load files do not included in OnlyPlot mode')
    #        sys.exit()
    #    if load_file!=[]:
    #        m_model.ModelStored = []
    #        for fil in load_file:
    #            print(fil)
    #            m_model.RestartModel()
    #            m_model.model.load_model(fil)
    #            m_model.AppendModel()
    #    print('Paso por 1')
    elif options.mode =='Load':
        print(" Saving the BDT("+str(options.signal)+" - "+str(options.channel)+") scores")
        print("  Output folder: "+str(options.output))
        ###Load the BDT result, the function add a new branch with the BDT result
        if options.signal == 'tHq':
            load_file = load_tHq
        elif options.signal == 'ttbar':
            load_file = load_ttbar
        elif options.signal == 'ttW':
            load_file = load_ttW
        elif options.signal == 'ttW_Sherpa':
            load_file = load_ttWSherpa
        elif options.signal == 'Diboson':
            load_file = load_VV
        elif options.signal == 'tZq':
            load_file = load_tZq
        elif options.signal == 'tWH':
            load_file = load_tWH
        elif options.signal == 'tH':
            load_file = load_tH
        else:
            print('ERROR: Load files do not included in Load mode')
            sys.exit()

        os.system('mkdir -p %s' % options.output)
        print (m_model.BDT_var)
        LoadModel(m_model, options.inputdir, tree_name, output = options.output, input_file=load_file, thread=True)

        # Check
        if len([name for name in os.listdir(options.inputdir) if os.path.isfile(os.path.join(options.inputdir, name))]) != len([name for name in os.listdir(options.output) if os.path.isfile(os.path.join(options.output, name))]):
            print("Different  number of files in the input and output directories ")

        
    else:
        print('ERROR: execution mode not recognize')
        sys.exit()
