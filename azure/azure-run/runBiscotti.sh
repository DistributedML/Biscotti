#!/bin/bash

IFS=$'\n'
set -f

# Argument parsing and intializing variables
#############################################################################
let indexCount=0
let dimensions=0
let nodesInEachVM=$1
let totalNodes=$2
hostFileName=$3
dataset=$4
additionalArgs=$5

echo $additionalArgs

let azure=1

# dataset="mnist"

#TODO: Take as input dataset name. Figure out dimensions based on name
if [[ "$dataset" = "mnist" ]]; then
	dimensions=7850
elif [["$dataset" = "creditcard"]]; then
	dimensions=25
fi

currentDir=$PWD
azureUser="matheus"
pathToKeyGeneration=$GOPATH/src/Biscotti/keyGeneration/
confPath=$GOPATH/src/Biscotti/azure/azure-conf/
distSysPath=$GOPATH/src/Biscotti/DistSys/
controllerIP=$(ifconfig | grep -oE -m 1 "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1)
controllerUser=$USER
logFilesPath=$distSysPath/LogFiles
hostPath="$confPath$hostFileName"
peersFile="peersFileSent"
firstport=8000
##############################################################################



# #Generate files to bootstrap clients with
# ###############################################################################

# Generate commitment and pk/sk
cd $pathToKeyGeneration
go install
$GOPATH/bin/keyGeneration -n=$nodesInEachVM -d=$dimensions -h=$hostFileName

# Build biscotti
cd $distSysPath
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




# #Transfer bootstrap files and biscotti binary to all hosts
# ##############################################################################################

for line in $(cat $hostPath);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo $tname

	# if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
	# 	azureUser="clement"
	# else
	# 	azureUser="cfung"
	# fi

	scp $distSysPath/commitKey.json $azureUser@$tname:$distSysPath
	scp $distSysPath/pKeyG1.json $azureUser@$tname:$distSysPath
	scp peersFileSent $azureUser@$tname:$distSysPath
	scp $distSysPath/DistSys $azureUser@$tname:$distSysPath

done
# ##################################################################################################

# #SSH into each host and run the required number of nodes in each host
# ###################################################################################################
for line in $(cat $hostPath);do

	echo "Here"
	echo deploying nodes in $line

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	if [[ "$azure" -eq 1 ]]; then
		echo "Deploying on azure"
		ssh $azureUser@$tname 'bash -s' < deployAzureNodes.sh $nodesInEachVM $indexCount $totalNodes $tname $controllerUser $controllerIP $logFilesPath $dataset $additionalArgs &
	else
		ssh $controllerUser@$controllerIP 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $controllerIP $logFilesPath&
	fi

	indexCount=$((indexCount + nodesInEachVM))

	# Uncomment if only needs to test one node deployment
	# break

done

wait

exit
###############################################################################################################


#Deprecated code
####################################################
	# skip the first line
	# if [ $cnt -eq 0 ]; then
	# 	cnt=$((cnt + 1))
	# 	continue
	# fi


	#naur
	# if [[ "$tname" == "198.162.52.126" ]]; then
	# 	ssh matheus@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &
	
	# # dijkstra 
	# elif [[ "$tname" == "198.162.52.154" ]]; then
	# 	bash deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &			

	# else

	# 	if [[ "$tname" == "198.162.52.157" || "$tname" == "198.162.52.33" ]]; then
	# 		$username="clement"
	# 	else
	# 		$username="cfung"
	# 	fi

	# 	if [[ "$azure" -eq 1 ]]; then
	# 		echo "Deploying on azure"
	# 		ssh $$azureUser@$tname 'bash -s' < deployAzureNodes.sh $nodesInEachVM $indexCount $totalNodes $tname $controllerUser $controllerIP $logFilesPath&	
	# 	else
	# 		ssh $$azureUser@$tname 'bash -s' < deployNodes.sh $nodesInEachVM $indexCount $totalNodes $tname &	
	# 	fi

	# fi
#################################################