#!/bin/bash

IFS=$'\n'
set -f

let indexCount=0
let nodesInEachVM=$1
let totalNodes=$2

for line in $(cat tempHosts);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	if [ "$tname" == "198.162.52.126" ]; then
		ssh shayan@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &			
	else
		ssh cfung@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &
	fi

	indexCount=$((indexCount + nodesInEachVM))
	
	# Give time for nodes in the firstVM to get bootstrapped

	if [ $indexCount -eq $nodesInEachVM ]; then
		echo "Sleeping. Allowing first set of nodes to get bootstrapped"
		sleep 10
	fi

done