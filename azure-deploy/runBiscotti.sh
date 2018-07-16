#!/bin/bash

IFS=$'\n'
set -f

let indexCount=0
let nodesInEachVM=10

for line in $(cat tempHosts);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	if [ "$tname" == "198.162.52.126" ]; then
		ssh shayan@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount 20 $tname &			
	else
		ssh cfung@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount 20 $tname &
	fi

	indexCount=$indexCount+$nodesInEachVM

	indexCount=$((indexCount + nodesInEachVM))
	
	# Give time for nodes in the firstVM to get bootstrapped

	if [ $indexCount -eq $nodesInEachVM ]; then
		echo "Sleeping. Allowing first set of nodes to get bootstrapped"
		sleep 7
	fi

done