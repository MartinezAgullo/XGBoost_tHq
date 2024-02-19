from root_numpy import root2array, rec2array, array2root
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from keras.optimizers import SGD

model = load_model('../KerasTMVA/model.h5')
#model.summary()


import matplotlib
matplotlib.use('Agg') # Bypass the need to install Tkinter GUI framework                                                                                                                                                                                                                                                    
import matplotlib.pyplot as plt

from data_file_DSID import DSID_LIST

FileName_refit = "/lhome/ific/j/jguerrero/KerasExample/mc16a.346230.aMCPy8EG_tHjb125_4fl_lep.AFII.nominal.root"
FileName_default = FileName_refit

TreeName_default = "tHqLoop_nominal"
#TreeName_refit = "Refit1_SiAndTRT"                                                                                                                                                                                                                                                                                         
#branches_t = ["PtLeptonMin","DeltaRMin","DelataPhiSS","MaxEtaJetNoBjet","DeltaEtaForwardLjetLeadingBjet","DeltaEtaForwardLjetClosestLepto","LeptonChargeSum"]
branches_t = ["MaxEtaJetNoBjet","LeptonChargeSum", "DeltaEtaForwardLjetLeadingBjet","DeltaEtaForwardLjetClosestLepto", "DeltaRMin", "DelataPhiSS","PtLeptonMin","m_met","m_njets","m_nbjets"]
input_dim = 10
array_input_sig = np.empty([1,input_dim])
array_input_bkg = np.empty([1,input_dim])
output = np.empty([1,1])
i=0
sig = 0
FILE = ['mc16e.346345.PhPy8EG_ttH125_2l.FS.nominal.root','mc16a.346230.aMCPy8EG_tHjb125_4fl_lep.AFII.nominal.root']#,'mc16e.346345.PhPy8EG_ttH125_2l.FS.nominal.root']
for file in os.listdir('../TopLoop/run_baseline/'):
    file_path = '../TopLoop/run_baseline/%s'%file
    dsid_temp = file.split('.')[1]

##################################3
    try:
        if dsid_temp in DSID_LIST['Review']:
#            print('File number: %s'%i)                                                                                                                                                                                                                                                                                     
#            i+=1                                                                                                                                                                                                                                                                                                           
#            if i>5 and id_temp not in DSID_LIST['tHq']: continue                                                                                                                                                                                                                                                           
            array_aux = root2array(file_path,TreeName_default,branches_t)
            array_aux = rec2array(array_aux)
#            print(array_aux[1,:])                                                                                                                                                                                                                                                                                         
#            print(array_input)                                                                                                                                                                                                                                                                                           
#            array_input = np.append(array_input, array_aux,axis=0)
#            print('Aux array')                                                                                                                                                                                                                                                                                             
#            print(array_aux.shape)                                                                                                                                                                                                                                                                                         
#            print('Internal array')                                                                                                                                                                                                                                                                                        
#            print(array_input.shape)                                                                                                                                                                                                                                                                                       

            if dsid_temp in DSID_LIST['tHq']:
                array_input_sig = np.append(array_input_sig, array_aux,axis=0)
                sig +=1
#                print(output)                                                                                                                                                                                                                                                                                              
                output = np.append(output, np.ones((array_aux.shape[0], 1)),axis = 0)
            else:
                i+=1
                array_input_bkg = np.append(array_input_bkg, array_aux,axis=0)
                output = np.append(output, np.zeros((array_aux.shape[0], 1)),axis = 0)
#            print('Output array')                                                                                                                                                                                                                                                                                          
#            print(output.shape)                                                                                                                                                                                                                                                                                            
        else:
            continue
    except:
        print(dsid_temp)
        print(file_path)


#array_input_sig= np.append(array_input, output, axis = 1)
array_input_sig=np.delete(array_input_sig,0,0)

#array_input= np.append(array_input, output, axis = 1)
array_input_bkg=np.delete(array_input_bkg,0,0)

"""
print(array_input_sig.shape)

np.random.seed(7)
np.random.shuffle(array_input)
trainX, testX, trainy, testy = train_test_split(array_input[:,:-1],array_input[:, -1],test_size = 0.1)
#score = model.evaluate(testX,testy, verbose=0)

#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#ns_probs = [0 for _ in range(len(testy))]
"""
print(array_input_sig.shape)
print(array_input_bkg.shape)
lr_probs_sig = model.predict_proba(array_input_sig)
lr_probs_bkg = model.predict_proba(array_input_bkg)
"""
#print(lr_probs)
#lr_probs = lr_probs[:, 0]

#ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
#print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('NN: ROC AUC=%.3f' % (lr_auc))


#ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

#plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.savefig('Roc_curve.png')
"""
plt.hist([lr_probs_sig,lr_probs_bkg], bins = 200,normed = True)#, label = 'Signal')
#plt.hist(lr_probs_bkg, bins = 20, label = 'Background')
plt.savefig('proof.png') 
