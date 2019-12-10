let converged=0
let churnRate=$1
dataset=$2
let numberOfNodes=$3

sleepAfter=$((60/$churnRate))

mkdir results

echo $sleepAfter
echo $dataset
echo $numberOfNodes

bisLocalPath="$GOPATH/src/Biscotti/DistSys/"

cd $bisLocalPath 
mkdir LogFiles

bash localTest.sh $numberOfNodes $dataset &

# Allow all clients to come online
sleep 15

while [[ "$converged" -eq 0 ]]; do

# 	#statements

	let failNode=$((( RANDOM % numberOfNodes)))
	
	let vmCount=0
	let failPort=$((failNode + 8000))
	
	echo $failNode
	echo $failPort

	if [[ "$failNode" -eq 0 ]]; then
		echo "I will not kill 0"
		continue
	fi
         
        echo "killing node"             
	fuser -k $failPort/tcp
	let failAfter=$(($sleepAfter - 5))
	echo $failAfter 			
	sleep $failAfter

	 echo "launching node"
        ./DistSys -i $failNode -t $numberOfNodes -d $dataset >> /dev/null &
        sleep 5

done


