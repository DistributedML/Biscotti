#!/bin/bash

IFS=$'\n'
set -f

# Argument parsing and intializing variables
#############################################################################
let indexCount=0
let nodesInEachVM=$1
let totalNodes=$2
hostFileName=$3

let dimensions=7850
let azure=1

azureUser="matheus"
currentDir=$PWD
#hostFileName="hosts_sameDC"
pathToKeyGeneration=$GOPATH/src/Biscotti/keyGeneration/
confPath=$GOPATH/src/Biscotti/azure/azure-conf/
FedSysPath=$GOPATH/src/Biscotti/FedSys
distSysPath=$GOPATH/src/Biscotti/DistSys
controllerIP=$(ifconfig | grep -oE -m 1 "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1)
controllerUser=$USER
logFilesPath=$FedSysPath/LogFiles
hostPath="$confPath$hostFileName"
peersFile="peersFileSent"
firstport=8000

#if [[ "$dataset" = "mnist" ]]; then
#	dimensions=7850
#elif [["$dataset" = "creditcard"]]; then
#	dimensions=25
#fi

##############################################################################



# #Generate files to bootstrap clients with
# ###############################################################################

# Build biscotti
cd $FedSysPath
echo "Building"
go build

# Come back to current folder
cd $currentDir

# Generate a file containing list of the peers and the respective ports they are running on.
#Used to bootstrap each node with list of peers

# Empty the file
> $peersFile

for line in $(cat $hostPath);do
	
	for (( port = $firstport; port < $firstport + $nodesInEachVM; port++ )); do
		echo "$line":"$port" >> $peersFile
	done

	firstport=$((firstport + nodesInEachVM))

done

# ##################################################################################




#Transfer bootstrap files and biscotti binary to all hosts
##############################################################################################

for line in $(cat $hostPath);do

	# skip the first line
	# if [ $cnt -eq 0 ]; then
	# 	cnt=$((cnt + 1))
	# 	continue
	# fi

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo $tname

	if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
		username="matheus"
	else
		username="matheus"
	fi

	scp peersFileSent $username@$tname:~/gopath/src/Biscotti/FedSys
	scp $FedSysPath/FedSys $username@$tname:~/gopath/src/Biscotti/FedSys

done
##################################################################################################



# #SSH into each host and run the required number of nodes in each host
# ###################################################################################################
for line in $(cat $hostPath);do

	echo "Here"
	echo deploying nodes in $line

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	#naur
	if [[ "$tname" == "198.162.52.126" ]]; then
		ssh $azureUser@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &
	
	# dijkstra 
	elif [[ "$tname" == "198.162.52.154" ]]; then
		bash deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &			

	else

		if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
			username="matheus"
		else
			username="matheus"
		fi

		if [[ "$azure" -eq 1 ]]; then
			echo "Deploying on azure"
			ssh $username@$tname 'bash -s' < deployAzureNodes.sh $nodesInEachVM $indexCount $totalNodes $tname $controllerUser $controllerIP $logFilesPath&	
		else
			ssh $username@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &	
		fi

	fi

	indexCount=$((indexCount + nodesInEachVM))

	# break

done

wait

exit
# ##############################################################################################################