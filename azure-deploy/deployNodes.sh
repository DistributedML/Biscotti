#!/bin/sh

nodesToRun=$1
startingIndex=$2
totalnodes=$3
myAddress=$4:

source ~/.profile

cd $GOPATH/src/simpleBlockChain/DistSys
go install

git pull origin master

myPrivateIp=$(ifconfig | grep -oE -m 1 "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1)
# myPrivateIp=$myPrivateIP+":"

echo $myPrivateIp

for (( index = startingIndex ; index < startingIndex + nodesToRun; index++ )); do
	
	thisLogFile=test1_$index\_$totalNodes.log
	
	let thisPort=8000+$index
	echo $myAddress
	echo $thisPort
	echo $index
	echo $totalnodes

	$GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard -f=peersfile.txt -a=$myAddress -p=$thisPort -pa=$myPrivateIp > $thisLogFile &
	
	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> outLog.log &
	if [ $index -eq 0 ] 
	then			
		sleep 5			
	fi
	break

done	

# wait

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

# done
