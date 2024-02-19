import os,sys,random
from xgboost.sklearn import XGBClassifier, Booster,DMatrix#, plot_importance, DMatrix 
#from keras.models import load_model
from time import sleep
import threading
import pandas as pd
import numpy as np
import array
from ROOT import TFile, TTree
from root_numpy import root2array
sys.path.append('../../tHqUtils/')
from Reader import  OpenYaml, Array_input_root, Branch2numpy

from ToolsMVA import ModuleKfold

from optparse import OptionParser
def UpdateTree(File,tree,branches):
	try:
		array_input =root2array(File,tree,branches)
	#array_input = Branch2numpy(branches_t,m_tree)
	except:
		print('The file %s is not included'%File)
		return
	print('Read: %s'%file)
	
	columns = branches
#
	df = pd.DataFrame(array_input, columns=columns)
	
	df.query('m_nNoBjets>=1 and m_nbjets>=1 and trilepton_mass_Bjet_l1<10000 and trilepton_mass_Bjet_l2<10000 and abs(LeptonChargeSum)==1',inplace=True)
	df.query('weight_nominal<200',inplace=True)
	mean_weight = df['weight_nominal'].mean()
	Data = False
	
	if file.split('.')[1]=='periodAllYear': Data = True
	BDT_responde = []
#		

	i = 0
	#for row in range(df.shape[0]):
	
	for index, row in df.iterrows():
		
		if Data:
			number_of_model = random.randint(0,4)
			weight = np.array([1.])
		else:
			try:
				number_of_model = row['eventNumber']%5
				weight = row['weight_nominal']/mean_weight
			except:
				print('error')
				print(df.shape[0])
				#print(df.ix[:,'eventNumber'])
				print(row)
				print('test')
				print(row['m_nNoBjets'])
				#print(df.ix[:row,'eventNumber'][0])
				#print(df.ix[row,'eventNumber'])
		#BDT_responde.append(eval('model_%i.predict_proba(df.ix[row,BDT_var])[:, 1][0]'%(df.ix[row,'eventNumber']%5)))
		BDT_responde.append(eval('model_%i.predict_proba(row[BDT_var])[:,1]'%number_of_model))
		#print(row[BDT_var].values.T)
		#dpred = DMatrix(row[BDT_var].values.reshape(1,-1), missing=-999.0, feature_names=BDT_var,weight=weight.reshape(1,-1))
		#BDT_responde.append(eval('model_%i.predict(dpred)'%number_of_model))
		i = i + 1
		status = float(i)/float(df.shape[0])*100
		if options.batch:
			#print('BATCH OPTION')
			if status==100:
			#	sys.stdout.write('\r')
				sys.stdout.write("[%-100s] %.3f%%(%i/%i)" % ('='*int(status), status,i,df.shape[0]))
			#	sys.stdout.flush()
		else:
			sys.stdout.write('\r')
			sys.stdout.write("[%-100s] %.3f%%(%i/%i)" % ('='*int(status), status,i,df.shape[0]))
			sys.stdout.flush()

	print('BDT responde finished')
	out_file = TFile(dirName+'/'+file,'recreate')
	new_tree = m_tree.CloneTree() 

	bdt_tHq = array.array('f', [0.0])
	branch = new_tree.Branch("bdt_tHq", bdt_tHq, 'bdt_tHq/F')
	print('start to fill new branch')
	
	for i in BDT_responde:
		bdt_tHq[0] = i 
		branch.Fill()
	print('Write root file')
	new_tree.Write()
	out_file.Close()
	#except:
	#	print('the file %s is not included'%file)


parser = OptionParser(usage = "usage: %prog arguments", version="%prog")
parser.add_option("--batch", dest="batch", action="store_true")

(options,args) = parser.parse_args()

if options.batch:
	print('Batch option')
#Create a folder for output files

dirName = 'out_test/'
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")


print('va por aqui')
param = {'max_depth': 4
             ,'objective': 'binary:logistic'
             ,'learning_rate':0.01
             ,'n_estimators':1000                                                                                     
             ,'min_child_weight' :0.04 
             #,'max_delta_step': 2                                                                                
             #,'eval_metric': ['error',"auc","logloss"]
             ,'n_jobs':-1
             #,'scale_pos_weight':(bkg_yield/sig_yield)/2
             
             }
model_0 = XGBClassifier(**param)
#model_0 = Booster(param)
model_1 = XGBClassifier(**param)
#model_1 = Booster(param)
model_2 = XGBClassifier(**param)
#model_2 = Booster(param)
model_3 = XGBClassifier(**param)
#model_3 = Booster(param)
model_4 = XGBClassifier(**param)
#model_4 = Booster(param)
print('Se va a cargar el modelo')
model_0.load_model('/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/XGBoost/model_LogLoss_K0.json')
model_1.load_model('/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/XGBoost/model_LogLoss_K1.json')
model_2.load_model('/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/XGBoost/model_LogLoss_K2.json')
model_3.load_model('/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/XGBoost/model_LogLoss_K3.json')
model_4.load_model('/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/XGBoost/model_LogLoss_K4.json')
print('Modelo cargado')

cfg = OpenYaml("/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/Utils/config.yaml")
 
# Variables from the analysis that are going to be used 
CfgParameter = cfg['ShortMVA']
BDT_var=[]
for var in CfgParameter.values():
    BDT_var.append(str(var['Name']))
branches_t =[]
branches_t+=BDT_var
branches_t.append('eventNumber')
branches_t.append('weight_nominal')

branches_t.append('m_nbjets')

tree_t = 'tHqLoop_nominal_Loose'
#file_path = '/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/out_baseline/mc16e.346799.aMCPy8EG_tHjb125_4fl_CPalpha_0_ML_2L.AFII.nominal.root'
#dir_path = '/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/out_baseline/mc16d.364250.Sh222_llll.FS.nominal.root'
#dir_path ='/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/out_baseline/mc16e.346799.aMCPy8EG_tHjb125_4fl_CPalpha_0_ML_2L.AFII.nominal.root'
#output_file_name = 'mc16e.346799.aMCPy8EG_tHjb125_4fl_CPalpha_0_ML_2L.AFII.nominal.root'
#dir_path = '/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/out_btag_eff_60_0/'
dir_path = '/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/out_eff_60_0_AllTight/'
stop = 3000
i=0
threads = []
for file in os.listdir(dir_path):
	#if file.split('.')[1]=='periodAllYear':continue

	if os.path.isfile(dirName+'/'+file): continue
#if True:
#	try:
	#file = dir_path
	print('Start to clone file %s'%file)
	file_path = dir_path + file
	
	m_file = TFile.Open(file_path)
	m_tree = m_file.Get(tree_t)
	if options.batch:
		print('por aqui')
		t = threading.Thread(target=UpdateTree, args=(file_path,tree_t,branches_t,))
		threads.append(t)
		t.start()
	else:
		UpdateTree(File=file_path,tree=tree_t,branches=branches_t)
	i = i +1

	
	
