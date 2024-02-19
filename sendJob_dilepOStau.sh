#!/bin/bash
cd /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA
source setupROOT.sh
source venv/bin/activate
# run mva_runner.py
echo "running: "$*
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonn/OS_Valencia_Structure/nominal_Loose/ -s ttbar --mode Train --name Test_1_TrueBonnSamples

# tH
#python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024/nominal_OS/ -s tH --mode Optimize --name test5_UseGA_I

python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -i/lustre/ific.uv.es/grid/atlas/t3/cescobar/tHq_analysis/13TeV/EBreview_v34_2L1tau_PLIV_SFs_syst_BDTassignment_lep3_pt_min14GeV/


# ttbar
python mva_runner.py -m XGBoost --channel OS --configfile /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024/nominal_OS/ -s tH --mode Train --name _FinalModel

deactivate
