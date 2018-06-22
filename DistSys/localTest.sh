#!/bin/sh

clear

# go install


# echo "Running tests: No failure case. All nodes online"

# #---------------------------------------------------Test 1: All nodes online--------------------------------------------------------------------

# for (( totalnodes = 2; totalnodes < 5; totalnodes++ )); do
	
# 	echo "Running with " $totalnodes "nodes"

# 	for (( index = 0; index < totalnodes; index++ )); do
		
# 		thisLogFile=test_$index\_$totalnodes.txt
# 		sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile & 
		
# 		if [ $index -eq 0 ] 
# 		then			
# 			sleep 10			
# 		fi
# 	done	

# 	wait
# 	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"
	
# 	for (( outernode = 0; outernode < totalnodes; outernode++ )); do	
# 		outerLogFile=test_$outernode\_$totalnodes.txt
# 		for (( innernode = 0; innernode < totalnodes; innernode++ )); do
# 			innerLogFile=test_$outernode\_$totalnodes.txt			
# 			if [ $innerLogFile=$outerLogFile ]; then
# 				continue
# 			fi
# 			if !(cmp -s $innerLogFile $outerLogFile) 
# 			then
# 				echo Test failed: $innerLogFile and $outerLogFile are different
# 				exit -1	
# 			fi		
# 		done
# 	done

# done

#---------------------------------------------------Test 1: All nodes online---------------------------------------------------------------------------

echo "Running tests: One node offline all the time"

for (( totalnodes = 3; totalnodes < 5; totalnodes++ )); do
	
	echo "Running with " $totalnodes "nodes"

	for (( index = 0; index < (totalnodes - 1); index++ )); do
		
		thisLogFile=test_$index\_$totalnodes.txt
		sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile & 
		
		if [ $index -eq 0 ] 
		then			
			sleep 10			
		fi
	done	

	wait
	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"
	
	for (( outernode = 0; outernode < totalnodes - 1; outernode++ )); do	
		outerLogFile=test_$outernode\_$totalnodes.txt
		for (( innernode = 0; innernode < totalnodes - 1; innernode++ )); do
			innerLogFile=test_$outernode\_$totalnodes.txt			
			if [ $innerLogFile=$outerLogFile ]; then
				continue
			fi
			if !(cmp -s $innerLogFile $outerLogFile) 
			then
				echo Test failed: $innerLogFile and $outerLogFile are different
				exit -1	
			fi		
		done
	done

done
