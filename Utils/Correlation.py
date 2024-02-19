import os, sys
from os.path import isfile
import numpy as np

import matplotlib
matplotlib.use('Agg') # Bypass the need to install Tkinter GUI framework                                                                                                                                                                                                        
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix

import seaborn as sns

# Include Reader file located in Utils directory
sys.path.append('../../tHqUtils/')
from Reader import Root2Df, OpenYaml #Root2DfInfo

##################################################
##################################################
##################################################

# ===========================
# value_to_color
# =========================== 
def value_to_color(values):
    # Use 256 colors for the diverging color palette
    n_colors = 256 
    # Create the palette
    palette = sns.diverging_palette(20, 220, n=n_colors) 
    # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    color_min, color_max = [-1, 1]
    out_pal = []
    for val in values:
        # position of value in the input range, relative to the length of the input range
        val_position = float((val - color_min)) / (color_max - color_min) 
        # target index in the color palette
        ind = int(val_position * (n_colors - 1)) 
        out_pal.append(palette[ind])
    return out_pal

# ===========================
# heatmap
# =========================== 
def heatmap(x, y, size,color):
    
    color_min, color_max = [-1, 1]
    n_colors = 256
    palette = sns.diverging_palette(20, 220, n=n_colors)
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 18.5)
    
    # Setup a 1x15 grid
    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.2) 
    ax = plt.subplot(plot_grid[:,:-1])
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique(),reverse=False)]
    y_labels = [v for v in sorted(y.unique(),reverse=True)]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    #Dot per inch
    dpi = 100 
    
    size_scale = (fig.dpi*fig.get_figwidth()*(ax.figure.subplotpars.right-ax.figure.subplotpars.left)*0.5)/float(len(x_labels))


    ax.scatter(
        x=x.map(x_to_num),      # Use mapping for x
        y=y.map(y_to_num),      # Use mapping for y
        c = color,
        s=size * size_scale**2, # Vector of square sizes, proportional to size parameter
        marker='s'              # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right',fontsize=9.5)
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels,fontsize=9.5)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # Use the rightmost column of the plot
    ax = plt.subplot(plot_grid[:,-1]) 

    # Fixed x coordinate for the bars
    col_x = [0]*len(palette) 
    
    # y coordinates for each of the n_colors bars
    bar_y=np.linspace(color_min, color_max,len(palette))
    bar_height = bar_y[1] - bar_y[0]
   
    ax.barh(
        y=bar_y,
        width=[5]*len(palette), # Make bars 5 units wide
        left=col_x, # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )

    ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.set_ylim(bar_y[0], bar_y[-1])
    ax.grid(False) # Hide grid
    ax.set_facecolor('white') # Make background white
    ax.set_xticks([]) # Remove horizontal ticks
    ax.set_yticks(np.round(np.linspace(min(bar_y), max(bar_y), 10),2)) # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right() # Show vertical ticks on the right 
# ===========================
# ScatterMatrix
# =========================== 
def ScatterMatrix(df, colors = False):

    sm = scatter_matrix(df, alpha=0.2)
    if colors:
        for i in range(sm.shape[0]):
            for j in range(sm.shape[1]):
                if abs(corr.iloc[i][j]) >= 0.66:
                    sm[i,j].set_facecolor('xkcd:red')
                elif 0.33 < abs(corr.iloc[i][j]) < 0.66:
                    sm[i,j].set_facecolor('xkcd:orange')
                elif abs(corr.iloc[i][j]) <= 0.33:
                    sm[i,j].set_facecolor('xkcd:green')
                else:
                    print("Bin %i,%i has correlation %f"%(i,j,corr.iloc[i][j]))
    plt.savefig('Scatter_matrix.pdf')
    plt.close()

# ===========================
# ScatterPlot
# =========================== 
def ScatterPlot(i,j,df):

    columns = df.columns
    if(columns[i]=='S/B' or columns[j]=='S/B'): return
    print('Correlation: %s VS %s'%(columns[i],columns[j]))
    sns.lmplot( x=columns[i], y=columns[j], data=df, fit_reg=False, hue='S/B', legend=True)
    
    plt.savefig("Corr_%s_VS_%s_2LOSTau_ttbar.pdf"%(columns[i],columns[j]))
    print('Figure Corr_%s_VS_%s_2LOSTau_ttbar.pdf saved'%(columns[i],columns[j]))
    plt.close()
## ===========================
## CorrelationNumber: Correlation with numerical values
## ==========================
def CorrelationNumber(corr,label,font_size,name=''):
    fig, ax = plt.subplots(tight_layout =True)
    fig.set_size_inches(18.5, 15.5)
    ax = sns.heatmap(corr, annot=True,fmt='.2f',cmap = sns.diverging_palette(20, 220, n=200),vmin=-1, vmax=1, center=0)
    ax.set_xticklabels(label, rotation=45, horizontalalignment='right',fontsize=font_size)
    ax.set_yticklabels(label, rotation=0, horizontalalignment='right',fontsize=font_size)
    plt.savefig("correlation_number_%s.png"%name)
    plt.savefig("correlation_number_%s.pdf"%name)
    plt.close()

## ===========================
## CorrelationColour: Correlation with degradeted colours 
## ==========================
def CorrelationColour(corr,label,font_size,name=''):
    fig, ax = plt.subplots(tight_layout =True)
    fig.set_size_inches(18.5, 15.5)
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(label, rotation=45, horizontalalignment='right',fontsize=font_size)
    ax.set_yticklabels(label, rotation=0,horizontalalignment='right',fontsize=font_size)
    plt.savefig("correlation_colour_%s.png"%name)
    plt.close()
## ===========================
## Correlation with degradeted colours and different size of squares
## ==========================
def CorrelationSquares(corr,label,font_size,name=''):
    corr.columns = label
    corr.index = label
    corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs(),
        color = value_to_color(corr['value'])
    )
    plt.savefig("correlation_squares_%s.png"%name)
    plt.savefig("correlation_squares_%s.pdf"%name)
    plt.close()
#####################################################################
channel = 'OS'      # Options: SS or Os
process = 'tH'   # Options: tH, ttbar or tHq
#Read config file
#cfg = OpenYaml("/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/Utils/config.yaml")
#cfg = OpenYaml( "/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/Utils/config_2LSS.yaml")
if channel == 'OS':
    cfg = OpenYaml( "/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml")
if channel == 'SS':
    cfg = OpenYaml( "/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_SS.yaml")
#cfg = OpenYaml( "/lhome/ific/j/jguerrero/tHqIFIC/tHqMVA/Utils/config_3L.yaml")

# Variables from the analysis that are going to be used 
#CfgParameter = cfg['ShortMVAttbar']
#CfgParameter = cfg['AllBDT']
#CfgParameter = cfg['ShortMVAttbar']
#CfgParameter = cfg['ShortMVAttW']
#CfgParameter = cfg['ShortMVAtHq']
if process == 'tHq': CfgParameter = cfg['baseline_tHq']
if process == 'ttbar': CfgParameter = cfg['ShortMVAttbar']
if process == 'tH': CfgParameter = cfg['ShortMVAtH'] 
if process not in ['tH', 'ttbar', 'tHq']:
    print("Selectect proper process")
    print("Exiting")
    exit()
if channel not in ['SS', 'OS']:
    print("Selectect proper channel (SS/OS)")
    print("Exiting")
#CfgParameter = cfg['ShortMVAVV']
#CfgParameter = cfg['ShortMVAtZq']
#branches_t=[]
#for var in CfgParameter.values():
#    branches_t.append(str(var['Name']))


cfg_dict = {}
for var in CfgParameter.values():
    if 'Label_name' in var.keys():
        cfg_dict[str(var['Name'])]=str(var['Label_name'])
    else:
        cfg_dict[str(var['Name'])]=str(var['Name'])
branches_t = cfg_dict.keys()
branches_t.append('mcChannelNumber')
branches_t.append('OS_LepHad')
label = cfg_dict.values()
tree_t = 'tHqLoop_nominal_Loose'

#Read root file from a directory, the path starts form the current XGBoost directory 
# An additional "column" is added to the array to store the training data  
#df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/V34_AtLeast1Bjet_AtLeast1jet_2Tight_2LSS/nominal_Loose/', TreeName = tree_t ,branches = branches_t, signal = 'tHq')
#df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/V34_AtLeast1Bjet_AtLeast1jet_2Tight_2LSS_BDT_VV/nominal_Loose/', TreeName = tree_t ,branches = branches_t)
#df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dileSStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/nominal_Loose/', TreeName = tree_t ,branches = branches_t)  
#df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepSStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/nominal_Loose/', TreeName = tree_t ,branches = branches_t)
#df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/nominal_Loose/', TreeName = tree_t ,branches = branches_t)

#df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/nominal_Loose/', TreeName = tree_t ,branches = branches_t)

if channel == 'OS':
    df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024/nominal_OS/',TreeName = tree_t ,branches = branches_t)
if channel == 'SS':
    df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024/nominal_SS',TreeName = tree_t ,branches = branches_t)

#df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/nominal_Loose/', TreeName = tree_t ,branches = branches_t)
#df= Root2Df('/lustre/ific.uv.es/grid/atlas/t3/jguerrero/tHq_analysis/13TeV/V34_AtLeast1Bjet_AtLeast1jet_3Tight_3L_BDT_ttW/nominal_Loose/', TreeName = tree_t ,branches = branches_t, exclude='AllYear')


if process == 'tHq':
    df_signal = df.query('mcChannelNumber==346799')
    df_bkg = df.query('mcChannelNumber!=346799') # tHq
if process == 'ttbar':
    df_signal = df.query('mcChannelNumber==410470')
    df_bkg = df.query('mcChannelNumber!=410470') # ttbar
if process == 'tH':
    df_signal = df.query('mcChannelNumber in [346799, 508776]')
    df_bkg = df.query('mcChannelNumber not in [346799, 508776]') # tHq + tWH

df_bkg = df_bkg.drop('mcChannelNumber',axis=1)
df_signal = df_signal.drop('mcChannelNumber',axis=1)
if channel == 'OS':
    df_signal = df_signal.query('OS_LepHad==1')
    df_bkg = df_bkg.query('OS_LepHad==1')
if channel == 'SS':
    df_signal = df_signal.query('OS_LepHad==0')
    df_bkg = df_bkg.query('OS_LepHad==0') 


#Scale values of variables
#For only signal
#df = df[df['S/B']==0]
#For only background
#df = df[df['S/B']==1]
#df = df.reindex(sorted(df.columns), axis=1)

##Create correlation matrix with values from DataFrame
#corr = df.drop(['S/B','DSID','NEvent'],axis=1).corr()
corr_signal = df_signal.corr()
corr_signal.fillna(0,inplace= True)

corr_bkg = df_bkg.corr()
corr_bkg.fillna(0,inplace= True)

font_size = 10
##df_copy = df[corr.columns]
##df_copy['S/B'] = df['S/B']
#
## ===========================
## Correlation with numerical values
## ==========================

if process == 'ttbar' and channel == 'OS':
    CorrelationNumber(corr_signal,label,font_size,'2LOSTau_BDT_Target_ttbar')
    CorrelationNumber(corr_bkg,label,font_size,'2LOSTau_BDT_OthersThan_ttbar')
if process == 'tHq' and channel == 'OS':
    CorrelationNumber(corr_signal,label,font_size,'2LOSTau_BDT_Target_tHq')
    CorrelationNumber(corr_bkg,label,font_size,'2LOSTau_BDT_OthersThan_tHq')
if process == 'tH' and channel == 'OS':
    CorrelationNumber(corr_signal,label,font_size,'2LOSTau_BDT_Target_tH')
    CorrelationNumber(corr_bkg,label,font_size,'2LOSTau_BDT_OthersThan_tH')


if process == 'tHq' and channel == 'SS':
    CorrelationNumber(corr_signal,label,font_size,'2LSSTau_BDT_Target_tHq')
    CorrelationNumber(corr_bkg,label,font_size,'2LSSTau_BDT_OthersThan_tHq')
if process == 'tH' and channel == 'SS':
    CorrelationNumber(corr_signal,label,font_size,'2LSSTau_BDT_Target_tH')
    CorrelationNumber(corr_bkg,label,font_size,'2LSSTau_BDT_OthersThan_tH')

## ===========================
## Correlation with degradeted colours 
## ==========================

#CorrelationColour(corr,label,font_size,'test')

## ===========================
## Correlation with degradeted colours and different size of squares
## ==========================

#CorrelationSquares(corr,label,font_size,'2LSS_BDT_VV')

#if True:
#    for i in range(corr.shape[0]):
#        for j in range(i+1,corr.shape[1]):
#            ScatterPlot(i,j,df)

