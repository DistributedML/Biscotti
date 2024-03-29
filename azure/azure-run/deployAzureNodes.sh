#!/bin/sh

# Arg passing and variable initialization
###################################################
nodesToRun=$1
startingIndex=$2
totalnodes=$3
myAddress=$4
controllerUser=$5
controllerIP=$6
logFileCopyPath=$7
dataset=$8
additionalArgs=${@:9}

argList=($additionalArgs)

rm -rf LogFiles

source ~/.profile
pathToBinary=$GOPATH/src/Biscotti/DistSys
#####################################################


cd $pathToBinary

# Single command that kills any previous process
pkill DistSys

# Remove previous logfiles
rm -r LogFiles

# Get private IP
myPrivateIp=$(ifconfig | grep -oE -m 1 "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1)

echo $PWD
mkdir -p LogFiles

hostindex=0


for (( index = $startingIndex ; index < $startingIndex + nodesToRun; index++ )); do
	
	thisLogFile=test1_$index\_$totalnodes.log
	thatLogFile=log_$index\_$totalnodes.log
	
	let thisPort=8000+$index

	echo deploying "$index"
	cd $pathToBinary

	commandToRun="timeout 30000 ./DistSys -i=${index} -t=${totalnodes} -d=${dataset} -f=peersFileSent \
				-a=$myAddress -p=$thisPort -pa=$myPrivateIp ${argList[@]}"
	commandList=($commandToRun)
	"${commandList[@]}" > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile & 

done	

wait

echo "Copying files back to controller"
scp ./LogFiles/*.log $controllerUser@$controllerIP:$logFileCopyPath


# echo "Running with " $nodesToRun "nodes complete. Testing similarity of blockchain"

# for (( outernode = startingIndex; outernode < startingIndex + nodesToRun; outernode++ )); do	
	
# 	outerLogFile=test1_$outernode\_$totalnodes.log
# 	for (( innernode = startingIndex; innernode < startingIndex + nodesToRun ; innernode++ )); do
# 		innerLogFile=test1_$innernode\_$totalnodes.log			
# 		if [ $innerLogFile == $outerLogFile ]; then
# 			continue
# 		fi
# 		if !(cmp -s $innerLogFile $outerLogFile) 
# 		then
# 			echo Test failed: $innerLogFile and $outerLogFile are different
# 			echo "Files are different"
# 			exit -1	
# 		fi		
# 	done
# done

# echo "SUCCESS! Nodes have same blockchain"

# exit







#Deprecated code
################################################################

# echo "Pulling latest source code from github"

# git reset --hard
# git pull origin master

# # if [[ "$myAddress" != "198.162.52.154" ]]; then
	
# # 	echo "Copying files back to naur"
# # 	scp *.log cfung@198.162.52.154:/home/cfung/gopath/src/simpleBlockChain/DistSys/LogFiles

# # fi

# for line in $(cat $GOPATH/src/simpleBlockChain/azure-deploy/tempHosts);do

# 	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

# 	echo $tname

# 	for (( myIndex = hostindex ; myIndex < hostindex + nodesToRun; myIndex++)); do

# 		echo $myIndex
# 		let myPort=8000+$myIndex
# 		lineToWrite=$tname:$myPort
# 		echo $lineToWrite >> peersfile.txt
	
# 	done

# 	echo "I am here"

# 	hostindex=$((hostindex + nodesToRun))

# done

	# if [ $index -eq 0 ] 
	# then			
	# 	echo "Sleeping. Allowing node zero to be up and running"
	# 	sleep 5			
	# fi

# myPrivateIp=$myPrivateIP+":"

# echo $myPrivateIp
#################################################################
