#!/bin/bash
cd /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA
source setupROOT.sh
source venv/bin/activate
# run mva_runner.py
echo "running: "$*

# Add tHq nominal
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/nominal_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/nominal_Loose/

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/nominal_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/nominal_Loose/

# Add tHq alternative                                                                                                                                                            
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/alternative_sample/


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/alternative_sample/ --mode Load

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/alternative_sample/ -o  /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/alternative_sample/


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/alternative_sample/ -o a /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/alternative_sample/
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/alternative_sample/ 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/nominal_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/nominal_Loose/ 


# Add ttbar alternative
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/alternative_sample/ 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/alternative_sample/



# OLD SAMPLES

python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/alternative_sample/ 

python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar -i/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/alternative_sample/  --mode Load 

# Old Samples                                                                                                                                                                                              
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1down_Loose/ -t  tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -t  tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlat3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_PunchThrough_MC16__1down_Loose/ -t  tHqLoop_JET_PunchThrough_AFII__1down_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/Js/ET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_PunchThrough_MC16__1up_Loose/ -t  tHqLoop_JET_PunchThrough_AFII__1up_Loose


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1down_Loose/ -t  tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/JET_JER_DataVsMC_MC16__1up_Loose/ -t  tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/JET_PunchThrough_MC16__1down_Loose/ -t  tHqLoop_JET_PunchThrough_AFII__1down_Loose

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/JET_PunchThrough_MC16__1up_Loose/ -t  tHqLoop_JET_PunchThrough_AFII__1up_Loose




# Add tHq AlternativeSamples
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/alternative_sample/


#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/Alternative_for_ttbarNLOgen/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/alternative_sample/ 


# Add ttbar nominal
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/nominal_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/nominal_Loose/


# Add ttbar alternative
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/alternative_sample/


# Add tWH nominal
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tWH -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/nominal_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar_tWH/nominal_Loose/ --mode Load

# Add tWH alternative
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tWH -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar/alternative_sample/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq_ttbar_tWH/alternative_sample/ --mode Load



# Add AFII samples
#echo "Adding BDT(tHq) for systematics :: AFII samples"

#echo "JET_JER_DataVsMC_MC16__1down_PseudoData_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/AFII_Samples/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose 

#echo "JET_JER_DataVsMC_MC16__1up_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/AFII_Samples/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose

#echo "JET_JER_DataVsMC_MC16__1down_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/AFII_Samples/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_JER_DataVsMC_MC16__1down_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose

#echo "JET_JER_DataVsMC_MC16__1up_PseudoData_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/AFII_Samples/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose

#echo "JET_PunchThrough_MC16__1down_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/AFII_Samples/JET_PunchThrough_MC16__1down_Loose/  -a /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_PunchThrough_MC16__1down_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1down_Loose

#echo "JET_PunchThrough_MC16__1up_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tHq --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/AFII_Samples/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/EBreview_v34_dilepOStau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV_BDT_tHq/JET_PunchThrough_MC16__1up_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1up_Loose

#echo "Adding BDT(ttbar) for systematics :: AFII samples"


#echo "tH - JET_JER_DataVsMC_MC16__1down_PseudoData_Loose" 
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/JET_JER_DataVsMC_MC16__1down_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/JET_PunchThrough_MC16__1down_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1down_Loose 

#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s tH  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar_tH/JET_PunchThrough_MC16__1up_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1up_Loose 




#echo "ttbar - JET_JER_DataVsMC_MC16__1down_PseudoData_Loose" 
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_JER_DataVsMC_MC16__1down_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose 

#echo "JET_JER_DataVsMC_MC16__1up_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/JET_JER_DataVsMC_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_JER_DataVsMC_MC16__1up_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose 

#echo "JET_JER_DataVsMC_MC16__1up_PseudoData_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_JER_DataVsMC_MC16__1up_PseudoData_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose 

#echo "JET_JER_DataVsMC_MC16__1down_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/JET_JER_DataVsMC_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_JER_DataVsMC_MC16__1down_Loose/ -t tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose 

#echo "JET_PunchThrough_MC16__1down_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/JET_PunchThrough_MC16__1down_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_PunchThrough_MC16__1down_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1down_Loose 

#echo "JET_PunchThrough_AFII__1up_Loose"
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -s ttbar  --mode Load -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/dilepOStau_2024/JET_PunchThrough_MC16__1up_Loose/ -o /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024_OS_bdt_ttbar/JET_PunchThrough_MC16__1up_Loose/ -t tHqLoop_JET_PunchThrough_AFII__1up_Loose 







deactivate
