#!/bin/bash

IFS=$'\n'

let indexCount=0
let nodesInEachVM=$1
let totalNodes=$2
let dimensions=$3
let azure=1

echo "file written"

# Generate keys
# cd ../keyGeneration
# go install
# $GOPATH/bin/keyGeneration -n=$nodesInEachVM -d=$dimensions

cd ../DistSys
echo "Building"
go build

# rm ./LogFiles2/*.log
# cd ..

cd ../eval_PerformanceNVM

peersFile="peersFileSent"
firstport=8000

# Empty the file
> $peersFile

let vmCount=0

for line in $(cat tempHosts);do
	
	if [[ "$vmCount" -ge 10  ]]; then
		
		nodesInEachVM=2
	fi
	for (( port = $firstport; port < $firstport + $nodesInEachVM; port++ )); do
		echo "$line":"$port" >> $peersFile
	done

	firstport=$((firstport + nodesInEachVM))
	vmCount=$((vmCount + 1))

done

# for line in $(cat tempHosts);do

# 	# skip the first line
# 	# if [ $cnt -eq 0 ]; then
# 	# 	cnt=$((cnt + 1))
# 	# 	continues
# 	# fi

# 	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

# 	echo $tname

# 	if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
# 		username="clement"
# 	else
# 		username="cfung"
# 	fi

# 	# scp ../DistSys/commitKey.json $username@$tname:/home/$username/gopath/src/simpleBlockChain/DistSys
# 	# scp ../DistSys/pKeyG1.json $username@$tname:/home/$username/gopath/src/simpleBlockChain/DistSys
# 	# scp peersFileSent $username@$tname:~/gopath/src/simpleBlockChain/DistSys
# 	scp ../DistSys/DistSys $username@$tname:~/gopath/src/simpleBlockChain/DistSys

# done

let myCount=0
let nodesInEachVM=3

for line in $(cat tempHosts);do

	if [[ "$myCount" -ge 10  ]]; then
		
		nodesInEachVM=2
	fi

	echo deploying nodes in $line

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	if [[ "$tname" == "198.162.52.126" ]]; then
		ssh shayan@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &			
	
	# dijkstra 
	elif [[ "$tname" == "198.162.52.154" ]]; then
		bash deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &			

	else

		if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
			username="clement"
		else
			username="cfung"
		fi

		if [[ "$azure" -eq 1 ]]; then
			echo "Deploying on azure"
			ssh $username@$tname 'bash -s' < deployAzureNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &	
		else
			ssh $username@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &	
		fi

	fi

	indexCount=$((indexCount + nodesInEachVM))
	myCount=$((myCount + 1))



done

wait

exit
