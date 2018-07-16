#!/bin/sh

nodesToRun=$1
startingIndex=$2
totalnodes=$3
myAddress=$4

source ~/.profile

cd $GOPATH/src/simpleBlockChain/DistSys

PID=`pgrep DistSys`
while sudo kill $PID > /dev/null
do
	sudo kill -9 $PID
	break
done

rm LogFiles

echo "Pulling latest source code from github"

git reset --hard
git pull origin master


echo "Compiling go"

go install

myPrivateIp=$(ifconfig | grep -oE -m 1 "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1)

# myPrivateIp=$myPrivateIP+":"

# echo $myPrivateIp

mkdir -p LogFiles

for (( index = startingIndex ; index < startingIndex + nodesToRun; index++ )); do
	
	thisLogFile=test1_$index\_$totalnodes.log
	thatLogFile=log_$index\_$totalnodes.log
	
	let thisPort=8000+$index

	sudo timeout 120 $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard -f=peersfile.txt -a=$myAddress -p=$thisPort -pa=$myAddress > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile &
	

	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard -f=peersfile.txt -a=$myAddress -p=$thisPort -pa=$myAddress > ./LogFiles/$thisLogFile 2> ./LogFiles/$thatLogFile &
	# sudo $GOPATH/bin/DistSys -i=$index -t=$totalnodes -d=creditcard > $thisLogFile 2> outLog.log &
	
	if [ $index -eq 0 ] 
	then			
		sleep 5			
	fi

done	

wait

cd ./LogFiles

if [[ "$myAddress" == "198.162.52.57" ]]; then
	
	echo "Copying files back to naur"

	scp *.log shayan@198.162.52.126:/home/shayan/gopath/src/simpleBlockChain/DistSys/LogFiles

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
