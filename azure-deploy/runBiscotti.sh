#!/bin/bash

IFS=$'\n'
set -f

let indexCount=0
let nodesInEachVM=10

for line in $(cat tempHosts);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	if [ "$tname" == "198.162.52.126" ]; then
		ssh shayan@$tname -x 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount 10 $tname &			
	else
		ssh cfung@$tname -x 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount 10 $tname &
	fi

	indexCount=$indexCount+$nodesInEachVM

done