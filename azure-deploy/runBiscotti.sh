#!/bin/bash

IFS=$'\n'
set -f

let indexCount=0
let nodesInEachVM=$1
let totalNodes=$2

peersFile="peersFileSent"
firstport=8000

# Empty the file
> $peersFile

for line in $(cat tempHosts);do
	
	for (( port = $firstport; port < $firstport + $nodesInEachVM; port++ )); do
		echo "$line":"$port" >> $peersFile
	done

	firstport=$((firstport + nodesInEachVM))

done

echo "file written"

for line in $(cat tempHosts);do

	echo deploying nodes in $line

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	if [ "$tname" == "198.162.52.126" ]; then
		ssh shayan@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &			
	
	elif [ "$tname" == "198.162.52.154" ]; then
		bash deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &			

	# leviathan
	elif [ "$tname" == "198.162.52.57" ]; then
		# echo $peersFile
		# scp peersFileSent cfung@$tname:~/gopath/src/simpleBlockChain/DistSys
		scp ../DistSys/DistSys cfung@$tname:~/gopath/src/simpleBlockChain/DistSys
		ssh cfung@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &			
	
	# emerson or clarke
	else		
		# echo $peersFile
		# scp peersFileSent clement@$tname:~/gopath/src/simpleBlockChain/DistSys
		scp ../DistSys/DistSys clement@$tname:~/gopath/src/simpleBlockChain/DistSys
		ssh clement@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &
	fi

	indexCount=$((indexCount + nodesInEachVM))
	
	# Give time for nodes in the firstVM to get bootstrapped

	if [ $indexCount -eq $nodesInEachVM ]; then
		echo "Sleeping. Allowing first set of nodes to get bootstrapped"
		sleep 15
	fi

done