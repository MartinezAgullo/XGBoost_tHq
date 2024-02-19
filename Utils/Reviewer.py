from ROOT import TFile, TTree
import os
from data_file_DSID import DSID_LIST
import numpy as np
#import pickle

review = []
f = open("data_file_DSID.py","wb")
for file in os.listdir('../TopLoop/run_baseline/'):
    entries = 0.
    file_path = '../TopLoop/run_baseline/%s'%file
    #file_path = '../TopLoop/run_baseline/mc16a.364115.Sh221_Zee_maxHtPtV0_70_C.FS.nominal.root'
    data_b = TFile.Open(file_path)
    background = data_b.Get('tHqLoop_nominal')
    entries = np.float64(background.GetEntries())

#    print('the file %s has %i  entries'%(file,entries))
    try:
        if entries  == 0. : 
            continue
    #elif entries ==1:
    #    print('Warning this file has only one entrie %s'%file)
    #    continue
        else:
            DSID_aux = file.split('.')[1]
            review.append(DSID_aux)
    except:
        print('WARNING: the file %s has %i entries and it has not been included'%(file,entries)) 
print review
DSID_LIST['Review'] = review
#DSID_LIST.update(Review = review)

f.write('DSID_LIST ='+str(DSID_LIST))
f.close()
#print DSID_LIST['Review']
"""
    if file.split('.')[1] in DSID_LIST['ttH']:
        data_s.append(TFile.Open(file_path))
        signal =  data_s[-1].Get('tHqLoop_nominal')
        dataloader.AddSignalTree(signal, 1.0)
        dataloader_1.AddSignalTree(signal, 1.0)
    else:
        data_b.append(TFile.Open(file_path))
        background = data_b[-1].Get('tHqLoop_nominal')
        if background.GetEntries() == 0 : continue
#        print background.GetEntries()                                                                                                                                                                                                                                                                                        
        dataloader.AddBackgroundTree(background, 1.0)
        dataloader_1.AddBackgroundTree(background, 1.0)
"""
