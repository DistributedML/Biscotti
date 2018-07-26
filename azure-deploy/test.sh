let nodesToRun=30
let startingIndex=0
let totalnodes=60

cd $GOPATH/src/simpleBlockChain/DistSys

rm peersfile.txt


let hostindex=0

for line in $(cat $GOPATH/src/simpleBlockChain/azure-deploy/tempHosts);do

	tname=`echo $line | grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'`

	echo $tname

	for (( myIndex = hostindex ; myIndex < hostindex + nodesToRun; myIndex++)); do

		echo $myIndex
		let myPort=8000+$myIndex
		lineToWrite=$tname:$myPort
		echo $lineToWrite >> peersfile.txt
	
	done

	echo "I am here"

	hostindex=$((hostindex + nodesToRun))

done

