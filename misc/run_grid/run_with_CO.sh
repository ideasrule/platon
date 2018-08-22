#!/bin/bash

GGCHEM_DIR=/home/stanley/packages/GGchem

for logZ in `seq -1 0.05 3`
do
    for CO in `seq 0.2 0.2 2.0`
    do
	trap "echo Exited!; exit;" SIGINT SIGTERM
	echo $logZ $CO
	
	dirname=result_"$logZ"_"$CO"
	rm -rf $dirname
	mkdir $dirname
	cd $dirname
	
	python ../makeAbun.py ../abund_solar.in $logZ $CO> abundances.in
	cp ../model_template.in model.in
	ln -s $GGCHEM_DIR/data data
		
	$GGCHEM_DIR/ggchem model.in > output_resume &
	cd ..
    done
    wait
done
