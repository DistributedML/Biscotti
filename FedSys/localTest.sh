#!/bin/sh

clear

#Flushing out all iptables blocked rules
sudo iptables -F INPUT
sudo iptables -F OUTPUT

let nodes=$1
let dimensions=$2
let cnt=0

# Single command that kills them
pkill FedSys
pkill DistSys

cd ../FedSys
go build

# Purge the logs
rm -f ./LogFiles/*.log

# #---------------------------------------------------Test 1: All nodes online--------------------------------------------------------------------

echo "Running tests: No failure case. All nodes online"

for (( totalnodes = $nodes; totalnodes < ($nodes + 1); totalnodes++ )); do
	
	echo "Running with" $totalnodes "nodes"

	for (( index = 0; index < totalnodes; index++ )); do
		
		thisLogFile=test1_$index\_$totalnodes.log
		thatLogFile=log_$index\_$totalnodes.log

		myAddress=127.0.0.1
		let thisPort=8000+$index
		echo $index
		echo $thisPort
		echo $myAddress

		./FedSys -i=$index -t=$totalnodes -d=mnist > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile & 
		
	done	

	wait

	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"
	
	cd LogFiles

	for (( outernode = 0; outernode < (totalnodes-1); outernode++ )); do	
		
		outerLogFile=test1_$outernode\_$totalnodes.log
		for (( innernode = 0; innernode < totalnodes ; innernode++ )); do
			innerLogFile=test1_$innernode\_$totalnodes.log			
			if [ $innerLogFile == $outerLogFile ]; then
				continue
			fi
			if !(cmp -s $innerLogFile $outerLogFile) 
			then
				echo Test failed: $innerLogFile and $outerLogFile are different
				echo "Files are different"
				exit -1	
			fi		
		done
	done

	echo "SUCCESS! Nodes have same blockchain"

done
		
