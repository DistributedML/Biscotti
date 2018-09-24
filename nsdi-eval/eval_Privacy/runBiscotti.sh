#!/bin/bash

IFS=$'\n'

let indexCount=0
let nodesInEachVM=$1
let totalNodes=$2
let dimensions=$3
let collusion=$4
let numberNoisers=$5
let azure=0

echo "file written"

# Generate keys
# cd ../keyGeneration
# go install
# $GOPATH/bin/keyGeneration -n=$nodesInEachVM -d=$dimensions

cd ../DistSys
echo "Building"
# go build

rm ./LogFiles/*.log
# cd ..

cd ../eval_Privacy

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

for line in $(cat tempHosts);do

	# skip the first line
	# if [ $cnt -eq 0 ]; then
	# 	cnt=$((cnt + 1))
	# 	continue
	# fi

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo $tname

	if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
		username="clement"
	else
		username="cfung"
	fi

	# scp ../DistSys/commitKey.json $username@$tname:/home/$username/gopath/src/simpleBlockChain/DistSys
	# scp ../DistSys/pKeyG1.json $username@$tname:/home/$username/gopath/src/simpleBlockChain/DistSys
	# scp peersFileSent $username@$tname:~/gopath/src/simpleBlockChain/DistSys
	# scp ../DistSys/DistSys $username@$tname:~/gopath/src/simpleBlockChain/DistSys

done

for line in $(cat tempHosts);do

	echo deploying nodes in $line

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	if [[ "$tname" == "198.162.52.126" ]]; then
		ssh shayan@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname $collusion $numberNoisers &			
	
	# dijkstra 
	elif [[ "$tname" == "198.162.52.154" ]]; then
		bash deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname $collusion $numberNoisers &			

	else

		if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
			username="clement"
		else
			username="cfung"
		fi

		if [[ "$azure" -eq 1 ]]; then
			echo "Deploying on azure"
			ssh $username@$tname 'bash -s' < deployAzureNodes.sh $nodesInEachVM $indexCount $totalNodes $tname $collusion $numberNoisers &	
		else
			ssh $username@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname $collusion $numberNoisers &	
		fi

	fi

	indexCount=$((indexCount + nodesInEachVM))

done

wait

exit
