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
from  Reader import Array_SgnBkg, OpenYaml

pop_size = 20

def rootDefaults() :

    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    
def main():
   
    from optparse import OptionParser
    parser = OptionParser(usage = "usage: %prog arguments", version="%prog")
    parser.add_option("-i","--idir",      dest="idir",                           help="configuration file (default: %default)")
    parser.add_option("-b","--bkg",       dest="bkg",                            help="configuration file (default: %default)")
    parser.add_option("-c","--config",    dest="config",                         help="configuration file (default: %default)")
    parser.add_option("-C","--channel",   dest="ochannel",                       help="configuration file (default: %default)")
    parser.add_option("-p","--parameter", dest="configparam")
    parser.add_option("--iteration", dest="iter")
    parser.set_defaults(config="config/mc16_13TeV_config.ini",idir="/lustre/ific.uv.es/grid/atlas/t3/jgarcian/Run2/config/fv1",ochannel="el",bkg="",configparam ='LeptonsCut',iter =100)
    (options,args) = parser.parse_args()
    
    rootDefaults()

    #Read config file
    cfg = OpenYaml("/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/GeneticAlgorithm/config.yaml")
    global option_g
    option_g = cfg['Options']


    CfgParameter = cfg[options.configparam]



    
    index = 0
    dbFiles = {}


    # Variables from the analysis that are going to be used 
    name_t=[]
    for var in CfgParameter.values():
        name_t.append(str(var['Name']))

    
    name_t.append("weight_nominal")
    name_t.append("m_nbjets")
    name_t.append("m_njets")
    branches_t = name_t

    tree_t = 'tHqLoop_nominal'

    #Read root file from a directory, the path starts form the current directory 
    # An additional "column" is added to the array to store the training data                                                                            

    array_input= Array_SgnBkg('/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/out/', TreeName = tree_t ,branches = branches_t)
    columns_t = branches_t                                                                                                                            
    columns_t.append('S/B') 
    dbFiles = pd.DataFrame(array_input, columns= columns_t)
    dbFiles.query("2>=m_nbjets>=1 and  3 >= m_njets >= 1",inplace=True)
    dbFiles.drop(['m_nbjets','m_njets'],axis=1,inplace=True)
    #Initial values
    #Scan for Background
    bkg_yield, bkg_error, bkg_raw =  get_yield(dbFiles[dbFiles['S/B']==0])
    sig_yield, sig_error, sig_raw =  get_yield(dbFiles[dbFiles['S/B']==1])
    sig = (sig_yield/np.sqrt(bkg_yield + sig_yield))
    sig_error = sig*np.sqrt(((1-sig_yield*(sig_yield+bkg_yield))/sig_yield)**2*sig_error**2+sig_error**2/(2*(sig_yield+bkg_yield))**2)
    sn = sig_yield/(bkg_yield+sig_yield)
    print('---------Initial values----------')
    print('Signal(%i):       %f +- %f'%(sig_raw,sig_yield,sig_error))
    print('Background(%i):    %f +- %f'%(bkg_raw,bkg_yield, bkg_error))
    print('Significance: %f +- %f'%(sig,sig_error))
    print('---------------------------------')
    #Description table for signal and background
    table_description(dbFiles[dbFiles['S/B']==0],name='bkg')
    table_description(dbFiles[dbFiles['S/B']==1],name='signal')

    # Generate initial population
    init_df = init_population(pop_size,CfgParameter)
    Zn_best = 0.
    output_df = pd.DataFrame(columns= init_df.columns)

    NStep = 0
    print('Number of total iter: %i'%int(options.iter))
    
    while NStep<int(options.iter):
        # Fitness function
        evaluate(init_df,dbFiles,CfgParameter)
        init_df.sort_values(by=['Zn'],inplace=True)
        init_df.reset_index(drop=True,inplace=True)

        # Selection
        Zn_condition = init_df.iloc[int(pop_size/2.)-1]['Zn']
        Zn_best = init_df.iloc[pop_size - 1]['Zn']

        # Print current best
        print("\n -> Best set of cuts for iter %i"%NStep)
        print(init_df.iloc[pop_size - 1])
        output_df.loc[NStep] = init_df.iloc[pop_size - 1]
        # Drop worst performing half
        init_df.drop(init_df.index[:int(pop_size/2)],inplace=True)
        init_df.reset_index(drop=True,inplace=True)
        #print 'TEST_1'
        #print init_df
        # Do crossover/mutation
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
            print('Moving on')
        else:
            init_df = init_df.append(init_population(missing,CfgParameter))
            init_df = init_df.reset_index(drop=True)
            #print 'TEST_3'
            #print init_df
            print("\n -> Added {} values".format(missing))

        print("\n -> Saved Results")
        print( init_df.query('Zn>0') )
        #init_df = init_df.append(mutate(init_df))
        NStep = NStep + 1
    #print(output_df)


    output_df.plot(subplots=True)

    plt.tight_layout()
    plt.savefig('output.png')
    plt.close()
    output_df.to_csv('output_GA_%s.csv'%options.configparam,index=False)
    sigcolumns =['SigYield','Sig','SN','Zn']
    try:
        EvolutionPlot(output_df.drop(sigcolumns,axis=1),2,3)
        plt.tight_layout()
        plt.savefig('evol_cuts.png')
        plt.close()
        EvolutionPlot(output_df[sigcolumns],2,2)
        plt.tight_layout()
        plt.savefig('evol_GA.png')
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
	            print('Last columns tryed %i'%n)
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
    sigcolumns =['SigYield','Sig','SN','Zn']
    
    for var in parameters.values():
        columns.append(str(var['Name']))
 

    dfcolumns = np.array(columns)
    dfcolumns = np.append(dfcolumns,sigcolumns)
    
    df = pd.DataFrame(columns=dfcolumns)

    dfvars = {}
#    print 'TEST'
#    print parameters.values()
    for i in range(pop_size):
        row = []

        for var in parameters.values():
            row.append(rand_value(float(var['Min']),float(var['Max']),int(var['Seed'])))

        row.append(0.000001)
        row.append(0.001)        
        row.append(-0.01)
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
            cut_aux_1 = cut_aux_1 + ' and '+extra+'('+var['Name']+option_g[var['Option']] +'{})'
        else:
            cut_aux_1 = extra+'('+var['Name']+option_g[var['Option']]+'{}) '

        if 'cut_aux_2' in locals():
            cut_aux_2 = cut_aux_2 + ',row[\''+var['Name']+'\']'
        else:
            cut_aux_2 = 'row[\''+var['Name']+'\']'
    aux_cut = '(\''+cut_aux_1+'\').format('+cut_aux_2+')'
#    print(aux_cut)
    return aux_cut
def evaluate(df,dbFiles,parameters):

    benchmark = 0
    aux_cut = StringCut(parameters)
#    print(aux_cut)
    for index, row in df.iterrows():

        #exec(aux_cut)
        cuts = eval(aux_cut)
        sig_yield, sig, sn, Zn1 = scan(cuts,dbFiles)
        row['SigYield'] = sig_yield
        row['Sig'] = sig
        row['SN'] = sn
        row['Zn'] = Zn1
#                row['SigYield'] = sig_yield[sigBenchmark]
#                row['Sig'] = sig[sigBenchmark]
#                row['SN'] = sn[sigBenchmark]
#                row['Zn'] = Zn1[sigBenchmark]
#                for x in range(1,len(signalDSIDs)) :
#                    row['SigYield'+"_"+str(signalDSIDs[x])] = sig[str(signalDSIDs[x])]
#                    row['Sig'+"_"+str(signalDSIDs[x])] = sig[str(signalDSIDs[x])]
#                    row['SN'+"_"+str(signalDSIDs[x])] = sn[str(signalDSIDs[x])]
#                    row['Zn'+"_"+str(signalDSIDs[x])] = Zn1[str(signalDSIDs[x])]


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


#    print total_sig_yield, total_bkg_yield+total_sig_yield['W+jets'],total_sig_yield['W+jets']/(total_bkg_yield+total_sig_yield['W+jets'])

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

 
