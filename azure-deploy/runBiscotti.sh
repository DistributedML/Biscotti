#!/bin/bash

IFS=$'\n'
set -f

for line in $(cat tempHosts);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`	
	# runScript=`deployNodes.sh 10 0 10 $tname`
	ssh nss@$tname -x 'bash -s' < deployNodes.sh 20 0 10 $tname &

done