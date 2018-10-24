#!/bin/sh

nodesToRun=$1
startingIndex=$2
totalnodes=$3
myAddress=$4

source ~/.profile

cd $GOPATH/src/simpleBlockChain/DistSys

# Single command that kills them
pkill DistSys

# echo "Pulling latest source code from github"

# git reset --hard
# git pull origin master

rm -r LogFiles

# echo "Compiling go"
# sudo go install
# stat $GOPATH/bin/DistSys

myPrivateIp=$(ifconfig | grep -oE -m 1 "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1)

# myPrivateIp=$myPrivateIP+":"

# echo $myPrivateIp

mkdir -p LogFiles

# create new peers file

# rm peersfile.txt

hostindex=0

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


for (( index = $startingIndex ; index < $startingIndex + nodesToRun; index++ )); do
	
	thisLogFile=test1_$index\_$totalnodes.log
	thatLogFile=log_$index\_$totalnodes.log
	

	let thisPort=8000+$index

	echo $index, $totalnodes, $myAddress, $thisPort


	echo deploying "$index"
	cd $GOPATH/src/simpleBlockChain/DistSys
	timeout 18000 ./DistSys -i=$index -t=$totalnodes \
		-d=mnist -f=peersFileSent \
		-a=$myAddress -p=$thisPort -pa=$myPrivateIp \
		 > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile &
	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard -f=peersfile.txt -a=$myAddress -p=$thisPort -pa=$myAddress > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile &
	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> outLog.log &

	# if [ $index -eq 0 ] 
	# then			
	# 	echo "Sleeping. Allowing node zero to be up and running"
	# 	sleep 5			
	# fi

done	

wait

cd ./LogFiles

if [[ "$myAddress" != "198.162.52.126" ]]; then
	
	echo "Copying files back to dijkstra"
	scp *.log shayan@198.162.52.126:/home/shayan/gopath/src/simpleBlockChain/DistSys/LogFiles2

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
