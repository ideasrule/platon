#!/bin/bash

for logZ in `seq -1 0.05 3`
do
    cd $1
    python ../read_ggchem.py $logZ ../../species_info
    cd ..
done
	    
