#!/bin/sh

clear

go install

echo "No failure case. All nodes online"

diffCommand='diff'

for (( totalnodes = 2; totalnodes < 3; totalnodes++ )); do
	
	echo "Running with " $totalnodes "nodes"

	for (( index = 0; index < totalnodes; index++ )); do
		
		thisLogFile=test_$index\_$totalnodes.txt
		diffCommand=$diffCommand' '$thisLogFile  
		echo $diffCommand
		sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile & 
		
		if [ $index -eq 0 ] 
		then			
			sleep 10			
		fi
	done	

	wait
	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"

	$diffCommand
	
done

