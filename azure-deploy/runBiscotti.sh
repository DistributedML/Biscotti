#!/bin/bash

IFS=$'\n'
set -f

let indexCount=0
let nodesInEachVM=10

for line in $(cat hosts);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`	
	# runScript=`deployNodes.sh 10 0 10 $tname`
	echo 
	ssh nss@$tname -x 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount 80 $tname &
	indexCount=$indexCount+$nodesInEachVM
	# break

done