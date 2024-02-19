#!/bin/bash
cd /lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/GeneticAlgorithm
source ../setupROOT.sh
source ../venv/bin/activate
python GeneticAlgorithmXGB.py  --channel OS -s tH --idir /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_2024/nominal_OS/
deactivate
