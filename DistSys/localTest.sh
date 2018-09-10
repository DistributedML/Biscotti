#!/bin/sh

clear

#Flushing out all iptables blocked rules

sudo iptables -F INPUT
sudo iptables -F OUTPUT


#Killing processes that might be using ports that I want

PID=`pgrep DistSys`
while sudo kill $PID > /dev/null
do
	sudo kill -9 $PID
	break
done

go install

cd LogFiles

rm *.log

cd ..

# #---------------------------------------------------Test 1: All nodes online--------------------------------------------------------------------

echo "Running tests: No failure case. All nodes online"

for (( totalnodes = 4; totalnodes < 5; totalnodes++ )); do
	
	echo "Running with " $totalnodes "nodes"

	for (( index = 0; index < totalnodes; index++ )); do
		
		thisLogFile=test1_$index\_$totalnodes.log
		thatLogFile=log_$index\_$totalnodes.log

		myAddress=127.0.0.1
		let thisPort=8000+$index
		echo $index
		echo $thisPort
		echo $myAddress

		sudo timeout 400 $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile & 
		# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> /dev/null &
		# if [ $index -eq 0 ] 
		# then			
		# 	sleep 10			
		# fi
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
		

# # #---------------------------------------------------Test 2: All nodes online except one---------------------------------------------------------------------------

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
	
# 	for (( outernode = 0; outernode < totalnodes-1; outernode++ )); do	
		
# 		outerLogFile=test2_$outernode\_$totalnodes.log
# 		for (( innernode = 0; innernode < totalnodes-1 ; innernode++ )); do
# 			innerLogFile=test2_$innernode\_$totalnodes.log			
# 			if [ $innerLogFile == $outerLogFile ]; then
# 				continue
# 			fi
# 			if !(cmp -s $innerLogFile $outerLogFile) 
# 			then
# 				echo Test failed: $innerLogFile and $outerLogFile are different
# 				echo "Files are different"
# 				exit -1	
# 			fi		
# 		done
# 	done

# 	echo "SUCCESS! Nodes have same blockchain"

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
# 			innerLogFile=test3_$innernode\_$totalnodes.log			
# 			if [ $innerLogFile == $outerLogFile ]; then
# 				continue
# 			fi
# 			if !(cmp -s $innerLogFile $outerLogFile) 
# 			then
# 				echo Test failed: $innerLogFile and $outerLogFile are different
# 				echo "Files are different"
# 				exit -1	
# 			fi		
# 		done
# 	done

# 	echo "SUCCESS! Nodes have same blockchain"

# done

# #---------------------------------------------------Test 4: One node fails and stay failed---------------------------------------------------------------------------

# echo "Running tests: Node joins, and then fails midway"


# for (( totalnodes = 3; totalnodes < 5; totalnodes++ )); do
	
# 	echo "Running with " $totalnodes "nodes"

# 	for (( index = 0; index < (totalnodes); index++ )); do
		
# 		thisLogFile=test4_$index\_$totalnodes.log
		
# 		# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile & 
# 		sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> /dev/null &
# 		sleep 10

# 	done

# 	echo "All nodes $totalnodes online. Causing last node to fail"

# 	let i=$totalnodes-1
# 	let PORT=8000+$i

# 	echo $PORT

# 	sudo iptables -A INPUT -p tcp --destination-port $PORT -j REJECT
# 	sudo iptables -A OUTPUT -p tcp --dport $PORT -j REJECT

# 	wait

# 	sudo iptables -F INPUT
# 	sudo iptables -F OUTPUT


# 	echo "Running with " $totalnodes "nodes complete. Testing similarity of blockchain"
	
# 	for (( outernode = 0; outernode < totalnodes-1; outernode++ )); do	
		
# 		outerLogFile=test4_$outernode\_$totalnodes.log
# 		for (( innernode = 0; innernode < totalnodes-1 ; innernode++ )); do
# 			innerLogFile=test4_$innernode\_$totalnodes.log			
# 			if [ $innerLogFile == $outerLogFile ]; then
# 				continue
# 			fi
# 			if !(cmp -s $innerLogFile $outerLogFile) 
# 			then
# 				echo Test failed: $innerLogFile and $outerLogFile are different
# 				echo "Files are different"
# 			fi		
# 		done
# 	done

# 	echo "SUCCESS! Nodes have same blockchain"

# done