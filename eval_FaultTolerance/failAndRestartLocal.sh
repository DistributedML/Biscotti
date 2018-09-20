let converged=0
let numberOfNodes=50
let failurePerMin=1

while [[ "$converged" -eq 0 ]]; do

	#statements
	let failNode=$((( RANDOM % numberOfNodes)))
	
	let vmCount=0
	let failPort=$((failNode + 8000))
	
	echo $failNode
	echo $failPort

	fuser -k $failPort/tcp
			
	sleep 5

done
