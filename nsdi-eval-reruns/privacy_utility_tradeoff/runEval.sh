#!/bin/sh

# Quick number of dimensions key:
# Creditcard => 25
# MNIST => 7850
# For others look into dataset.py in ML/Pytorch

let nodesInEachVM=$1
let totalNodes=$2
let numberOfRuns=$3
hostFileName=$4
dataset=$5
currentDir="$PWD"
additionalArgs="-sa=true -vp=true -np=false -na=3 -nv=3 -nn=2"

azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"

cd $azuredeployScript

for (( run = 0; run < $numberOfRuns; run++ )); do

	cd $azuredeployScript
	
	destFolder="$(currentDir)/LogFiles_$(run)"
	rm -rf $destFolder
	echo "Run#"$run
	bash runBiscotti.sh $nodesInEachVM $totalNodes $hostFileName $dataset "$additionalArgs"	
	# cp -a /home/shayan/gopath/src/Biscotti/DistSys/LogFiles $destFolder

done
