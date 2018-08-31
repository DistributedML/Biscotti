#!/bin/bash

#Generates bootstrap files at index 0 node and copies to the other nodes in the network
# CAN ONLY RUN THIS SCRIPT AT THE MACHINE IN THE FIRSTLINE OF THE HOSTS FILE

IFS=$'\n'
set -f

let nodesInEachVM=$1
let dimensions=$2
let cnt=0

username="cfung"

cd ../keyGeneration
go install

sudo $GOPATH/bin/keyGeneration -n=$nodesInEachVM -d=$dimensions

cd ../DistSys



for line in $(cat ../azure-deploy/tempHosts);do

	# skip the first line
	if [ $cnt -eq 0 ]; then
		cnt=$((cnt + 1))
		continue
	fi

	echo $line

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	scp commitKey.json $username@$tname:/home/$username/gopath/src/simpleBlockChain/DistSys
	scp pKeyG1.json $username@$tname:/home/$username/gopath/src/simpleBlockChain/DistSys
	scp peersfile.txt $username@$tname:/home/$username/gopath/src/simpleBlockChain/DistSys
	

done

