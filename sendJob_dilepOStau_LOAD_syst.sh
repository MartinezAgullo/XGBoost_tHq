#!/bin/bash
cd /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA
source setupROOT.sh
source venv/bin/activate
# run mva_runner.py
#echo "running: "$*

#echo "Adding BDT(tHq) for systematics"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/$*/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/$*/ -t tHqLoop_$*



#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/Test_ProblematicSample/JET_Pileup_PtTerm__1up_Loose/  -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_Pileup_PtTerm__1up_Loose/ -t tHqLoop_JET_Pileup_PtTerm__1up_Loose

#echo "Adding BDT(ttbar) for systematics"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/$*/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/$*/ -t tHqLoop_$*



#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/$*/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/$*/  -t tHqLoop_$*

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/$*/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/$*/ -t tHqLoop_$*


python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/$*/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/$*/  -t tHqLoop_$* 

python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/$*/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/$*/  -t tHqLoop_$*


#echo "Adding BDT(tWH) for systematics"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tWH --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/$*/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar_tWH/$*/ -t tHqLoop_$*    


#echo "Adding BDT(tHq) for systematics :: AFII samples"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1down_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_PunchThrough_MC16__1down_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII_1down_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_PunchThrough_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose
 



#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1down_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1down_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_PunchThrough_MC16__1up_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_PunchThrough_MC16__1down_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1down_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_PunchThrough_MC16__1up_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_PunchThrough_MC16__1down_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1down_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1down_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1down_Loose/ -t 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_PunchThrough_MC16__1up_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_PunchThrough_MC16__1down_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1down_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_PunchThrough_MC16__1up_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1up_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/JET_PunchThrough_MC16__1down_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1down_Loose





deactivate


