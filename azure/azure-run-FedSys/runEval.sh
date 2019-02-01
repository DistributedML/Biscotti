#!/bin/sh

# Quick number of dimensions key:
# Creditcard => 25
# MNIST => 7850
# For others look into dataset.py in ML/Pytorch

let nodesInEachVM=$1
let totalNodes=$2
let dimensions=$3
let numberOfRuns=$4

for (( run = 0; run < $numberOfRuns; run++ )); do
	
	destFolder=LogFiles_$run

	rm -rf $destFolder
	rm -rf /home/shayan/gopath/src/simpleBlockChain/FedSys/LogFiles/*.log 
	echo "Run#"$run
	bash runFedSys.sh $nodesInEachVM $totalNodes $dimensions $numberOfRuns	
	cp -a /home/shayan/gopath/src/simpleBlockChain/FedSys/LogFiles $destFolder

done
