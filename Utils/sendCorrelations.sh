#!/bin/bash                                                                                                                                 
source ../setupROOT.sh
virtualenv ../venv
source ../venv/bin/activate
python Correlation.py 


deactivate
