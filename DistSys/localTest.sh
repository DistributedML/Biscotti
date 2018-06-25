#!/bin/sh

# clear

# PID=`pgrep DistSys`
# while sudo kill $PID > /dev/null
# do
# 	sudo kill -9 $PID

# 	break
# done


# go install

# echo "Running tests: No failure case. All nodes online"

# #---------------------------------------------------Test 1: All nodes online--------------------------------------------------------------------

# for (( totalnodes = 2; totalnodes < 5; totalnodes++ )); do
	
# 	echo "Running with " $totalnodes "nodes"

# 	for (( index = 0; index < totalnodes; index++ )); do
		
# 		thisLogFile=test1_$index\_$totalnodes.log
# 		sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile & 
# 		# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> /dev/null &
# 		if [ $index -eq 0 ] 
# 		then			
# 			sleep 10			
# 		fi
# 	done	

# 	wait
# 	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"
	
# 	for (( outernode = 0; outernode < totalnodes; outernode++ )); do	
# 		outerLogFile=test1_$outernode\_$totalnodes.log
# 		for (( innernode = 0; innernode < totalnodes; innernode++ )); do
# 			innerLogFile=test1_$outernode\_$totalnodes.log			
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

# #---------------------------------------------------Test 2: All nodes online---------------------------------------------------------------------------

# echo "Running tests: One node offline all the time"

# for (( totalnodes = 3; totalnodes < 5; totalnodes++ )); do
	
# 	echo "Running with " $totalnodes "nodes"

# 	for (( index = 0; index < (totalnodes - 1); index++ )); do
		
# 		thisLogFile=test2_$index\_$totalnodes.log
# 		# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile & 
# 		sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> /dev/null &
# 		if [ $index -eq 0 ] 
# 		then			
# 			sleep 10			
# 		fi
# 	done	

# 	wait
# 	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"
	
# 	for (( outernode = 0; outernode < totalnodes - 1; outernode++ )); do	
# 		outerLogFile=test2_$outernode\_$totalnodes.log
# 		for (( innernode = 0; innernode < totalnodes - 1; innernode++ )); do
# 			innerLogFile=test2_$outernode\_$totalnodes.log			
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

# #---------------------------------------------------Test 3: Nodes join at later iterations---------------------------------------------------------------------------

# echo "Running tests: Node joins at a later iteration"

# for (( totalnodes = 3; totalnodes < 5; totalnodes++ )); do
	
# 	echo "Running with " $totalnodes "nodes"

# 	for (( index = 0; index < (totalnodes); index++ )); do
		
# 		thisLogFile=test3_$index\_$totalnodes.log
# 		# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile & 
# 		sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> /dev/null &
# 		if [ $index -le 1 ] 
# 		then
# 			sleep 10
# 		else
# 			sleep 50			
# 		fi
# 	done	

# 	wait
# 	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"
	
# 	for (( outernode = 0; outernode < totalnodes; outernode++ )); do	
# 		outerLogFile=test3_$outernode\_$totalnodes.log
# 		for (( innernode = 0; innernode < totalnodes ; innernode++ )); do
# 			innerLogFile=test3_$outernode\_$totalnodes.log			
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

#---------------------------------------------------Test 4: Nodes fail and come online---------------------------------------------------------------------------

echo "Running tests: Node joins, and then fails midway"

for (( totalnodes = 3; totalnodes < 5; totalnodes++ )); do
	
	echo "Running with " $totalnodes "nodes"

	for (( index = 0; index < (totalnodes); index++ )); do
		
		thisLogFile=test4_$index\_$totalnodes.log
		
		# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile & 
		sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> /dev/null &
		sleep 10

	done

	echo "All nodes $totalnodes online. Causing last node to fail"

	let i=$totalnodes-1
	let PORT=8000+$i

	echo $PORT

	sudo iptables -A INPUT -p tcp --destination-port $PORT -j DROP
	sudo iptables -A OUTPUT -p tcp --dport $PORT -j DROP

	sleep 15

	sudo iptables -F INPUT
	sudo iptables -F OUTPUT

	wait
	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"
	
	for (( outernode = 0; outernode < totalnodes; outernode++ )); do	
		outerLogFile=test4_$outernode\_$totalnodes.log
		for (( innernode = 0; innernode < totalnodes ; innernode++ )); do
			innerLogFile=test4_$outernode\_$totalnodes.log			
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
