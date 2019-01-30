#!/bin/sh

nodesToRun=$1
startingIndex=$2
totalnodes=$3
myAddress=$4

source ~/.profile

cd $GOPATH/src/Biscotti/DistSys

# Single command that kills them
pkill DistSys

rm -r LogFiles
mkdir -p LogFiles

myPrivateIp=$(ifconfig | grep -oE -m 1 "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1)

for (( index = $startingIndex ; index < $startingIndex + nodesToRun; index++ )); do
	
	thisLogFile=test1_$index\_$totalnodes.log
	thatLogFile=log_$index\_$totalnodes.log
	
	let thisPort=8000+$index

	echo deploying "$index" $myAddress $myPrivateIp
	cd $GOPATH/src/Biscotti/DistSys
	timeout 120 ./DistSys -i=$index -t=$totalnodes \
		-d=creditcard -f=peersFileSent \
		-a=$myAddress -p=$thisPort -pa=$myAddress \
		 > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile &

	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard -f=peersfile.txt -a=$myAddress -p=$thisPort -pa=$myAddress > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile &
	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> outLog.log &

	if [ $index -eq 0 ] 
	then			
		echo "Sleeping. Allowing node zero to be up and running"
		sleep 5			
	fi

done	

wait

cd ./LogFiles

if [[ "$myAddress" != "198.162.52.154" ]]; then
	
	echo "Copying files back to dijkstra"
	scp *.log cfung@198.162.52.154:/home/cfung/gopath/src/Biscotti/DistSys/LogFiles

fi

echo "Running with " $nodesToRun "nodes complete. Testing similarity of blockchain"

for (( outernode = startingIndex; outernode < startingIndex + nodesToRun; outernode++ )); do	
	
	outerLogFile=test1_$outernode\_$totalnodes.log
	for (( innernode = startingIndex; innernode < startingIndex + nodesToRun ; innernode++ )); do
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

exit
