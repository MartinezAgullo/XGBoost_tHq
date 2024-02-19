#!/bin/bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh --quiet
lsetup "root 6.10.04-x86_64-slc6-gcc62-opt" --quiet
export PYTHONPATH=$PYTHONPATH:/usr/lib64/python2.6/site-packages
