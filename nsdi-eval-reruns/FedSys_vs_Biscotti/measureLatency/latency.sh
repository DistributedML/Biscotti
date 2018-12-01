#!/bin/bash



confPath=$GOPATH/src/simpleBlockChain/azure-conf/
hostFileName="hosts_sameDC"
hostPath="$confPath$hostFileName"

# > privateIps

# for line in $(cat $hostPath);do
	
# 	# echo $line

# 	myPrivateIp='ifconfig | grep -oE -m 1 "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1'

# 	ssh cfung@$line $myPrivateIp >> privateIps

# done

# > pingValues

# let nodeCount=0

# for publicIp in $(cat $hostPath);do
	
# 	let startIndex=0
	
# 	echo $publicIp

# 	for privateIP in $(cat privateIps);do

# 		echo $privateIP

# 		if [[ $startIndex -le $nodeCount ]]; then
			
# 			startIndex=$((startIndex + 1)) 
# 			continue
		
# 		fi
	
# 		pingCommand='ping -c 1 '$privateIP'| grep rtt | cut -d"/" -f5'
		
# 		echo $pingCommand

# 		ssh cfung@$publicIp $pingCommand >> pingValues

# 		startIndex=$((startIndex + 1))

# 	done

# 	nodeCount=$((nodeCount + 1))		

	

# done

let minLatency=2000
let maxLatency=0

let totalCount=0
let totalLatency=0
let avgLatency=0

for latency in $(cat pingValues);do

	let thisLat=$latency
	echo $thisLat
	# totalCount=$((totalCount + 1))
	# totalLatency=$(($totalLatency + $thisLat))

	# if [[ $thisLat -lt minLatency ]]; then
	# 	#statements
	# 	minLatency=$latency
	# fi

	# if [[ $thisLat -gt maxLatency ]]; then
	# 	#statements
	# 	maxLatency=$latency
	# fi
done

avgLatency=$(($totalLatency/$totalCount)) 
echo $minLatency
echo $maxLatency
