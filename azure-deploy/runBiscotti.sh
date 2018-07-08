#!/bin/bash

IFS=$'\n'
set -f

let indexCount=0
let nodesInEachVM=10

for line in $(cat tempHosts);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`	
	ssh nss@$tname -x 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount 40 $tname &
	indexCount=$indexCount+$nodesInEachVM

done