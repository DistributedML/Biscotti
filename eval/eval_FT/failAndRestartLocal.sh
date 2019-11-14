let converged=0
let numberOfNodes=50

while [[ "$converged" -eq 0 ]]; do

	#statements
	let failNode=$((( RANDOM % numberOfNodes)))
	
	let vmCount=0
	let failPort=$((failNode + 8000))
	
	echo $failNode
	echo $failPort

	if [[ "$failNode" -eq 0 ]]; then
		echo I will not kill 0
		continue
	fi
         
    echo killing node 
            
	fuser -k $failPort/tcp
			
	sleep 5

	echo "launching node"

	./DistSys -i $failNode -t $numberOfNodes -d creditcard >> /dev/null &

	sleep 5

done
