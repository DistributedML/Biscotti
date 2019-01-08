#!/bin/sh

# Quick number of dimensions key:
# Creditcard => 25
# MNIST => 7850
# For others look into dataset.py in ML/Pytorch

if [ "$#" -ne 6 ]; then
    echo "Illegal number of parameters (expecting 2):"
    echo "[nodesInEachVM, totalNodes, numberOfRuns epsilon hostFileName dataset]"
    exit
fi

let nodesInEachVM=$1
let totalNodes=$2
let numberOfRuns=$3
epsilon=$4
hostFileName=$5
dataset=$6
currentDir=$PWD
additionalArgs="-sa=false -vp=true -np=false -na=3 -nv=3 -nn=2 -ep=$epsilon" # change epsilon value
echo $additionalArgs

azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"

cd $azuredeployScript

for (( run = 0; run < $numberOfRuns; run++ )); do

	cd $azuredeployScript	
	destFolder="$currentDir/LogFiles_$run"
	echo $destFolder
	# rm -rf $destFolder
	echo "Run#"$run
	bash runBiscotti.sh $nodesInEachVM $totalNodes $hostFileName $dataset "$additionalArgs"	
	cp -a /home/shayan/gopath/src/Biscotti/DistSys/LogFiles $destFolder

done
